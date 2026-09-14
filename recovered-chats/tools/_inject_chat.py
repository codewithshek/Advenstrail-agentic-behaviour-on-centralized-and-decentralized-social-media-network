"""Rebuild the recovered agent transcript as a real Cursor chat in the sidebar.

Reads the surviving .jsonl transcript and writes:
  composerHeaders                      -> one row (sidebar entry)
  cursorDiskKV composerData:<id>       -> conversation skeleton
  cursorDiskKV bubbleId:<id>:<bubble>  -> one row per message
"""
import json, os, re, sqlite3, uuid, datetime

SRC = r'C:\Users\dabhi\.cursor\projects\c-Users-dabhi-Documents-Major-Project\agent-transcripts\1ebbe10b-5f70-4dd8-a541-c3742a23f9da\1ebbe10b-5f70-4dd8-a541-c3742a23f9da.jsonl'
DB = os.path.join(os.environ['APPDATA'], 'Cursor', 'User', 'globalStorage', 'state.vscdb')

COMPOSER_ID = '1ebbe10b-5f70-4dd8-a541-c3742a23f9da'
NAME = 'Recovered: Adversarial Agentic AI project (Sep 9-11)'
WORKSPACE_ID = 'a5e6c1da0e5af66d47e7a3d1c2095889'
REPO = r'c:\Users\dabhi\Documents\Major-Project'

URI = {
    "$mid": 1,
    "fsPath": REPO,
    "_sep": 1,
    "external": "file:///c%3A/Users/dabhi/Documents/Major-Project",
    "path": "/C:/Users/dabhi/Documents/Major-Project",
    "scheme": "file",
}

EMPTY_CONTEXT = {
    "composers": [], "selectedCommits": [], "selectedPullRequests": [], "selectedImages": [],
    "selectedDocuments": [], "selectedVideos": [], "folderSelections": [], "fileSelections": [],
    "terminalFiles": [], "selections": [], "terminalSelections": [], "selectedDocs": [],
    "externalLinks": [], "cursorRules": [], "cursorCommands": [], "gitPRDiffSelections": [],
    "subagentSelections": [], "browserSelections": [], "extraContext": [],
    "mentions": {
        "composers": {}, "selectedCommits": {}, "selectedPullRequests": {}, "gitDiff": [],
        "gitDiffFromBranchToMain": [], "selectedImages": {}, "selectedDocuments": {},
        "selectedVideos": {}, "folderSelections": {}, "fileSelections": {}, "terminalFiles": {},
        "selections": {}, "terminalSelections": {}, "selectedDocs": {}, "externalLinks": {},
        "diffHistory": [], "cursorRules": {}, "cursorCommands": {}, "uiElementSelections": [],
        "consoleLogs": [], "ideEditorsState": [], "gitPRDiffSelections": {},
        "subagentSelections": {}, "browserSelections": {},
    },
}

EMPTY_LISTS = [
    'approximateLintErrors', 'lints', 'codebaseContextChunks', 'commits', 'pullRequests',
    'attachedCodeChunks', 'assistantSuggestedDiffs', 'gitDiffs', 'interpreterResults', 'images',
    'attachedFolders', 'attachedFoldersNew', 'userResponsesToSuggestedCodeBlocks',
    'suggestedCodeBlocks', 'diffsForCompressingFiles', 'relevantFiles', 'toolResults', 'notepads',
    'capabilities', 'multiFileLinterErrors', 'diffHistories', 'recentLocationsHistory',
    'recentlyViewedFiles', 'fileDiffTrajectories', 'docsReferences', 'webReferences',
    'aiWebSearchResults', 'attachedFoldersListDirResults', 'humanChanges', 'summarizedComposers',
    'cursorRules', 'cursorCommands', 'pastChats', 'contextPieces', 'editTrailContexts',
    'allThinkingBlocks', 'diffsSinceLastApply', 'deletedFiles', 'supportedTools',
    'attachedFileCodeChunksMetadataOnly', 'consoleLogs', 'uiElementPicked', 'knowledgeItems',
    'documentationSelections', 'externalLinks', 'projectLayouts', 'capabilityContexts', 'todos',
    'mcpDescriptors', 'workspaceUris',
]


def base_bubble(btype, bid, text, created_iso):
    b = {'_v': 3, 'type': btype, 'bubbleId': bid}
    for k in EMPTY_LISTS:
        b[k] = []
    b.update({
        'isAgentic': btype == 2,
        'existedSubsequentTerminalCommand': False,
        'existedPreviousTerminalCommand': False,
        'requestId': '',
        'attachedHumanChanges': False,
        'cursorCommandsExplicitlySet': False,
        'pastChatsExplicitlySet': False,
        'tokenCount': {'inputTokens': 0, 'outputTokens': 0},
        'isRefunded': False,
        'unifiedMode': 2,
        'createdAt': created_iso,
        'conversationState': '~',
        'text': text,
    })
    return b


def rich_text(text):
    paras = [p for p in text.split('\n')]
    content = []
    for p in paras:
        if p.strip():
            content.append({'type': 'paragraph', 'content': [{'type': 'text', 'text': p}]})
        else:
            content.append({'type': 'paragraph'})
    return json.dumps({'type': 'doc', 'content': content or [{'type': 'paragraph'}]})


def parse_user(text):
    m = re.search(r'<user_query>(.*?)</user_query>', text, re.S)
    body = m.group(1).strip() if m else text.strip()
    ts = re.search(r'<timestamp>(.*?)</timestamp>', text, re.S)
    stamp = None
    if ts:
        try:
            stamp = datetime.datetime.strptime(
                re.sub(r'\s*\(UTC[^)]*\)', '', ts.group(1)).strip(),
                '%A, %b %d, %Y, %I:%M %p')
            stamp = stamp - datetime.timedelta(hours=5, minutes=30)
        except Exception:
            stamp = None
    return stamp, body


# ---- read transcript -------------------------------------------------------
messages = []  # (role, text)
with open(SRC, 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        role = o.get('role')
        if role not in ('user', 'assistant'):
            continue
        content = (o.get('message') or {}).get('content') or []
        if isinstance(content, str):
            content = [{'type': 'text', 'text': content}]
        texts, tools = [], []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get('type') == 'text' and part.get('text'):
                texts.append(part['text'])
            elif part.get('type') == 'tool_use':
                args = part.get('input') or {}
                if isinstance(args, dict):
                    label = (args.get('description') or args.get('path') or args.get('command')
                             or args.get('pattern') or '')
                else:
                    label = str(args)
                tools.append('%s — %s' % (part.get('name', 'tool'), str(label)[:200]))
        if role == 'user':
            if not texts:
                continue
            stamp, body = parse_user('\n'.join(texts))
            if body:
                messages.append(('user', body, stamp))
        else:
            body = '\n\n'.join(t.strip() for t in texts if t.strip())
            if tools:
                block = '\n'.join('- `%s`' % t for t in tools)
                body = (body + '\n\n' + block).strip() if body else '**Tool calls**\n' + block
            if body:
                messages.append(('assistant', body, None))

print('messages to inject:', len(messages),
      '(user=%d)' % sum(1 for m in messages if m[0] == 'user'))

# ---- timeline --------------------------------------------------------------
start = datetime.datetime(2026, 9, 9, 13, 52, 0)
end = datetime.datetime(2026, 9, 11, 4, 40, 0)
span = (end - start).total_seconds()
cur = start
stamps = []
for i, (role, body, stamp) in enumerate(messages):
    if role == 'user' and stamp:
        cur = max(cur, stamp)
    else:
        cur = cur + datetime.timedelta(seconds=span / max(len(messages), 1) * 0.4)
    stamps.append(cur)

# ---- build bubbles ---------------------------------------------------------
headers, bubbles = [], []
for (role, body, _), when in zip(messages, stamps):
    bid = str(uuid.uuid4())
    iso = when.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    ms = int(when.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
    btype = 1 if role == 'user' else 2
    b = base_bubble(btype, bid, body, iso)
    b['context'] = json.loads(json.dumps(EMPTY_CONTEXT))
    if role == 'user':
        b['richText'] = rich_text(body)
        b['agentMode'] = 1
        b['modelInfo'] = {'modelName': 'auto-smart'}
        b['isPlanExecution'] = False
    else:
        b['startedAtMs'] = ms
    bubbles.append((bid, b))

    grouping = {'isRenderable': True, 'hasText': True,
                'textPreview': body[:300].replace('\n', ' '),
                'toolDisplayComputed': True}
    if role == 'user':
        grouping['routedModelLabel'] = 'Claude Opus 5'
    h = {'bubbleId': bid, 'type': btype, 'grouping': grouping,
         'contentHeightHint': 68, 'createdAt': iso}
    if role == 'assistant':
        h['startedAtMs'] = ms
    headers.append(h)

created_ms = int(start.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
updated_ms = int(end.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)

composer_data = {
    '_v': 18,
    'composerId': COMPOSER_ID,
    'richText': '',
    'text': '',
    'hasLoaded': True,
    'name': NAME,
    'fullConversationHeadersOnly': headers,
    'conversationMap': {},
    'status': 'completed',
    'context': json.loads(json.dumps(EMPTY_CONTEXT)),
    'generatingBubbleIds': [],
    'isReadingLongFile': False,
    'codeBlockData': {},
    'originalFileStates': {},
    'newlyCreatedFiles': [],
    'newlyCreatedFolders': [],
    'createdAt': created_ms,
    'lastUpdatedAt': updated_ms,
    'hasChangedContext': True,
    'activeTabsShouldBeReactive': True,
    'capabilities': [],
    'capabilityContexts': [],
    'isFileListExpanded': False,
    'unifiedMode': 'agent',
    'activeCustomMode': None,
    'committedCustomMode': None,
    'pendingExitedCustomMode': None,
    'forceMode': 'edit',
    'usageData': {},
    'allAttachedFileCodeChunksUris': [],
    'modelConfig': {'modelName': 'auto-smart', 'maxMode': False,
                    'selectedModels': [{'modelId': 'auto-smart', 'parameters': []}]},
    'subComposerIds': [],
    'subagentComposerIds': [],
    'todos': [],
    'isQueueExpanded': True,
    'queueItems': [],
    'hasUnreadMessages': False,
    'gitHubPromptDismissed': False,
    'totalLinesAdded': 0,
    'totalLinesRemoved': 0,
    'addedFiles': 0,
    'removedFiles': 0,
    'isDraft': False,
    'isAgentic': True,
    'isCreatingWorktree': False,
    'isApplyingWorktree': False,
    'isUndoingWorktree': False,
    'applied': False,
    'pendingCreateWorktree': False,
    'worktreeStartedReadOnly': False,
    'isBestOfNParent': False,
    'isBestOfNSubcomposer': False,
    'isContinuationInProgress': False,
    'isNAL': False,
    'isProject': False,
    'isSpec': False,
    'isSpecSubagentDone': False,
    'subtitle': 'Recovered after Cursor reinstall',
    'trackedGitRepos': [{'repoPath': REPO,
                         'branches': [{'branchName': 'main', 'lastInteractionAt': updated_ms}]}],
    'workspaceIdentifier': {'id': WORKSPACE_ID, 'uri': URI},
}

head = {
    'type': 'head',
    'composerId': COMPOSER_ID,
    'name': NAME,
    'createdAt': created_ms,
    'lastUpdatedAt': updated_ms,
    'unifiedMode': 'agent',
    'forceMode': 'edit',
    'hasUnreadMessages': False,
    'contextUsagePercent': 0,
    'totalLinesAdded': 0,
    'totalLinesRemoved': 0,
    'filesChangedCount': 0,
    'subtitle': 'Recovered after Cursor reinstall',
    'hasBlockingPendingActions': False,
    'hasPendingPlan': False,
    'isDraft': False,
    'isWorktree': False,
    'worktreeStartedReadOnly': False,
    'isSpec': False,
    'isProject': False,
    'isBestOfNSubcomposer': False,
    'numSubComposers': 0,
    'referencedPlans': [],
    'trackedGitRepos': [{'repoPath': REPO,
                         'branches': [{'branchName': 'main', 'lastInteractionAt': updated_ms}]}],
    'workspaceIdentifier': {'id': WORKSPACE_ID, 'uri': URI},
    'agentLocation': {'type': 'local',
                      'environment': {'id': WORKSPACE_ID, 'uri': URI}, 'status': 'active'},
    'agentLocationHistory': [],
}

# ---- write -----------------------------------------------------------------
conn = sqlite3.connect(DB)
conn.execute('pragma busy_timeout=15000')
cur_db = conn.cursor()
cur_db.execute('delete from cursorDiskKV where key like ?', ('bubbleId:' + COMPOSER_ID + ':%',))
cur_db.execute('insert or replace into cursorDiskKV(key, value) values(?,?)',
               ('composerData:' + COMPOSER_ID, json.dumps(composer_data)))
for bid, b in bubbles:
    cur_db.execute('insert or replace into cursorDiskKV(key, value) values(?,?)',
                   ('bubbleId:%s:%s' % (COMPOSER_ID, bid), json.dumps(b)))
cur_db.execute(
    'insert or replace into composerHeaders'
    '(composerId, workspaceId, createdAt, lastUpdatedAt, isArchived, isSubagent, recency,'
    ' checkpointAt, subagentTypeName, value) values(?,?,?,?,?,?,?,?,?,?)',
    (COMPOSER_ID, WORKSPACE_ID, str(created_ms), str(updated_ms), 0, 0, str(updated_ms),
     None, '', json.dumps(head)))
conn.commit()
print('bubbles written:', len(bubbles))
print('header row:', cur_db.execute(
    'select composerId, workspaceId, recency from composerHeaders where composerId=?',
    (COMPOSER_ID,)).fetchone())
conn.close()
print('done')
