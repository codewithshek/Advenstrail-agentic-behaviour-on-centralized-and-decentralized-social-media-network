import sqlite3, os, json

DB = os.path.join(os.environ['APPDATA'], 'Cursor', 'User', 'globalStorage', 'state.vscdb')
REC = '1ebbe10b-5f70-4dd8-a541-c3742a23f9da'
c = sqlite3.connect('file:%s?mode=ro' % DB.replace('\\', '/'), uri=True)


def load(key):
    r = c.execute('select value from cursorDiskKV where key=?', (key,)).fetchone()
    if not r:
        return None
    v = r[0]
    if isinstance(v, bytes):
        v = v.decode('utf-8', 'replace')
    return json.loads(v)


# every composer with real bubbles, newest first
rows = c.execute('select composerId, workspaceId, recency from composerHeaders order by recency desc').fetchall()
live = None
for cid, ws, rec in rows:
    if cid == REC or cid == 'empty-state-draft':
        continue
    n = c.execute('select count(*) from cursorDiskKV where key like ?', ('bubbleId:%s:%%' % cid,)).fetchone()[0]
    print('%-40s ws=%-34s bubbles=%d' % (cid, str(ws)[:34], n))
    if n > 3 and live is None:
        live = cid

print('\nlive reference chat:', live)
rec = load('composerData:' + REC)
if not live:
    print('no live chat with bubbles to compare')
    raise SystemExit

liv = load('composerData:' + live)
print('\n_v  live=%s  recovered=%s' % (liv.get('_v'), rec.get('_v')))
missing = sorted(set(liv) - set(rec))
extra = sorted(set(rec) - set(liv))
print('\nKEYS IN LIVE BUT MISSING FROM RECOVERED (%d):' % len(missing))
for k in missing:
    val = liv[k]
    s = json.dumps(val)[:160] if not isinstance(val, str) else repr(val[:160])
    print('  %-45s %s' % (k, s))
print('\nKEYS ONLY IN RECOVERED:', extra)

print('\n--- conversationState / conversationMap ---')
for k in ('conversationState', 'conversationMap', 'status', 'isAgentic', 'unifiedMode'):
    print('  %-22s live=%s' % (k, json.dumps(liv.get(k))[:200]))
    print('  %-22s rec =%s' % ('', json.dumps(rec.get(k))[:200]))

print('\n--- bubble field diff (first assistant bubble) ---')
def first_bubble(cid, want):
    d = load('composerData:' + cid)
    for h in d['fullConversationHeadersOnly']:
        if h.get('type') == want:
            b = load('bubbleId:%s:%s' % (cid, h['bubbleId']))
            if b:
                return b
    return None

lb = first_bubble(live, 2)
rb = first_bubble(REC, 2)
if lb and rb:
    print('_v live=%s rec=%s' % (lb.get('_v'), rb.get('_v')))
    miss = sorted(set(lb) - set(rb))
    print('missing from recovered bubble (%d):' % len(miss))
    for k in miss:
        print('  %-40s %s' % (k, json.dumps(lb[k])[:140]))
