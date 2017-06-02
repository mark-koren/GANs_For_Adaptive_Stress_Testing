import rllab
import joblib
import numpy as np

def save_trials(iters, path, header):
    i = 0
    data = joblib.load(path + '/itr_' + str(i) + '.pkl')
    env = data['env']
    w_env = env.wrapped_env
    out = np.array(w_env.get_cache_list())
    for i in range(1, iters):
        data = joblib.load(path + '/itr_' + str(i) + '.pkl')
        env = data['env']
        w_env = env.wrapped_env
        iter_array = np.array(w_env.get_cache_list())
        iter_array[:, 0] = i
        out = np.concatenate((out, iter_array), axis=0)

    np.savetxt(fname=path+'/trials.csv',
               X=out,
               delimiter=',',
               header=header)





