from pipeline.source.depparse import dep_feat_equals

dep_feat = {'token': 'drive', 'compounds': []}
feat = 'hard Drive'
assert not dep_feat_equals(dep_feat, feat)

dep_feat['compounds'] = ['Hard']
assert dep_feat_equals(dep_feat, feat)
