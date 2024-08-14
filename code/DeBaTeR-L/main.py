from SELFRec import SELFRec
from util.conf import ModelConf
import argparse

if __name__ == '__main__':
    baseline = ['NCF','LightGCN','MF','NGCF','DirectAU']
    graph_models = ['SGL', 'SimGCL']
    data_augmentation = ['BOD']

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # print('Baseline Models:'W)
    # print('   '.join(baseline))
    # print('-' * 80)
    # print('Graph-Based Models:')
    # print('   '.join(graph_models))
    # print('-' * 80)
    # print('Denoising Models:')
    # print('   '.join(data_augmentation))

    # print('=' * 80)
    # model = input('Please enter the model you want to run:')
    model = 'BOD'
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=str, help='dataset name')
    args = parser.parse_args()

    s = time.time()
    if model in baseline or model in graph_models or model in data_augmentation:
        conf = ModelConf('./conf/' + model + '.conf')
        conf.config['training.set'] = './dataset/{}/train.txt'.format(args.d)
        conf.config['test.set'] = './dataset/{}/test.txt'.format(args.d)
        conf.config['temporal.data'] = './dataset/{}'.format(args.d)
        conf.config['dataset.name'] = args.d
        conf.config['batch_size'] = '64' if ((args.d == 'ml-100k') or (args.d == 'ml-100k-p'))  else '2048'
        conf.config['learnRate'] = '1e-4' if ((args.d == 'ml-100k') or (args.d == 'ml-100k-p')) else '1e-3'
        if args.d == 'ml-100k':
            conf.config['GM_AU'] = conf.config['GM_AU'].replace('generator_lr 0.001', 'generator_lr 0.0001')
    else:
        print('Wrong model name!')
        exit(-1)
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
