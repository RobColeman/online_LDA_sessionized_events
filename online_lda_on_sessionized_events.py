import cPickle, string, numpy, getopt, sys, random, time, re, pprint

import onlineldavb
import mongo_client

project_id = "id"

# The number of documents to analyze each k
batchsize = 500
# The total number of documents in Wikipedia
D = mongo_client.get_session_count(project_id)
# The number of topics
K = 6

vocab = mongo_client.get_events_ids_by_project_id(project_id)
W = len(vocab)
model = onlineldavb.OnlineLDA(vocab, K, D)

for k, (n_skip,n_limit) in enumerate(build_batches(n, batch_size)):

    sessions = mongo_client.get_sessions_batch(project_id, n_skip, n_limit)
        
    (gamma, bound) = model.update_lambda(sessions)

    (event_tokens, event_counts) = onlineldavb.parse_sessions_list(sessions, model._vocab)
    perwordbound = bound * len(sessions) / (D * sum(map(sum, event_counts)))

    print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
        (k, model._rhot, numpy.exp(-perwordbound))

    if (k % 10 == 0):
        numpy.savetxt('lambda-%d.dat' % k, model._lambda)
        numpy.savetxt('gamma-%d.dat' % k, gamma)
