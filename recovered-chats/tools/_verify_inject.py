import sqlite3, os, json

DB = os.path.join(os.environ['APPDATA'], 'Cursor', 'User', 'globalStorage', 'state.vscdb')
CID = '1ebbe10b-5f70-4dd8-a541-c3742a23f9da'
c = sqlite3.connect(DB)
print('integrity:', c.execute('pragma integrity_check').fetchone()[0])
print('journal mode:', c.execute('pragma journal_mode').fetchone()[0])

print('\nheaders in sidebar:')
for r in c.execute('select composerId, workspaceId, recency, isArchived from composerHeaders order by recency desc'):
    v = c.execute('select value from composerHeaders where composerId=?', (r[0],)).fetchone()[0]
    if isinstance(v, bytes):
        v = v.decode('utf-8', 'replace')
    nm = json.loads(v).get('name', '(unnamed)')
    print('  %-38s ws=%-34s %-14s %s' % (r[0][:38], str(r[1])[:34], r[2], nm))

v = c.execute('select value from cursorDiskKV where key=?', ('composerData:' + CID,)).fetchone()[0]
if isinstance(v, bytes):
    v = v.decode('utf-8', 'replace')
d = json.loads(v)
print('\ncomposerData parses OK; name=%r; headers=%d' % (d['name'], len(d['fullConversationHeadersOnly'])))
n = c.execute('select count(*) from cursorDiskKV where key like ?', ('bubbleId:' + CID + ':%',)).fetchone()[0]
print('bubble rows:', n)

first = d['fullConversationHeadersOnly'][0]
bid = first['bubbleId']
bv = c.execute('select value from cursorDiskKV where key=?', ('bubbleId:%s:%s' % (CID, bid),)).fetchone()[0]
if isinstance(bv, bytes):
    bv = bv.decode('utf-8', 'replace')
b = json.loads(bv)
print('\nfirst bubble type=%s createdAt=%s' % (b['type'], b['createdAt']))
print('text starts:', b['text'][:180].replace('\n', ' '))
c.close()
