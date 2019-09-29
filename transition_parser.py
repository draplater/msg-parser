from six.moves import range
from io import open

import tree_utils
from common_utils import set_proc_name
from conll_reader import CoNLLUSentence
import sh
import os
import pickle
import sys
import time
import transition_utils

from arc_hybrid import ArcHybridLSTM
from transition_cmd import get_parser


def get_sentences(file_name):
    with open(file_name) as f:
        return [tree_utils.Sentence.from_conllu_sentence(i)
                for i in CoNLLUSentence.get_all_sentences(f)]


def train_parser(options, sentences_train=None, sentences_dev=None, sentences_test=None):
    current_path = os.path.dirname(__file__)
    set_proc_name(options.title)
    if not (options.rlFlag or options.rlMostFlag or options.headFlag):
        print('You must use either --userlmost or --userl or --usehead (you can use multiple)')
        sys.exit()

    if not sentences_train:
        sentences_train = get_sentences(options.conll_train)
    if not sentences_dev:
        sentences_dev = get_sentences(options.conll_dev) \
            if options.conll_dev is not None else None
    if not sentences_test:
        sentences_test = get_sentences(options.conll_test) \
            if options.conll_test is not None else None

    print('Preparing vocab')
    words, w2i, pos, rels = tree_utils.vocab(sentences_train)
    if not os.path.exists(options.output):
        os.mkdir(options.output)
    with open(os.path.join(options.output, options.params), 'wb') as paramsfp:
        pickle.dump((words, w2i, pos, rels, options), paramsfp)
    print('Finished collecting vocab')
    print('Initializing blstm arc hybrid:')
    parser = ArcHybridLSTM(words, pos, rels, w2i, options)
    for epoch in range(options.epochs):
        print('Starting epoch', epoch)
        parser.Train(sentences_train)

        def predict(sentences, gold_file, output_file):

            with open(output_file, "w") as f:
                result = parser.Predict(sentences)
                for i in result:
                    f.write(i.to_string())

            eval_script = os.path.join(current_path, "utils/evaluation_script/conll17_ud_eval.py")
            weight_file = os.path.join(current_path, "utils/evaluation_script/weights.clas")
            eval_process = sh.python(eval_script, "-v", "-w", weight_file,
                                     gold_file, output_file, _out=output_file + '.txt')
            eval_process.wait()
            sh.cat(output_file + '.txt', _out=sys.stdout)

            print('Finished predicting {}'.format(gold_file))

        if sentences_dev:
            dev_output = os.path.join(options.output, 'dev_epoch_' + str(epoch + 1) + '.conllu')
            predict(sentences_dev, options.conll_dev, dev_output)

        if sentences_test:
            test_output = os.path.join(options.output, 'test_epoch_' + str(epoch + 1) + '.conllu')
            predict(sentences_test, options.conll_test, test_output)

        for i in range(epoch + 1 - options.max_model):
            filename = os.path.join(options.output, options.model + str(i))
            if os.path.exists(filename):
                os.remove(filename)
        parser.Save(os.path.join(options.output, options.model + str(epoch + 1)))


if __name__ == '__main__':
    parser = get_parser()
    (options, args) = parser.parse_args()
    print('Using external embedding:', options.external_embedding)

    current_path = os.path.dirname(__file__)
    set_proc_name(options.title)

    if not options.predictFlag:
        train_parser(options)
    else:
        with open(options.params, 'r') as paramsfp:
            words, w2i, pos, rels, stored_opt = pickle.load(paramsfp)

        stored_opt.external_embedding = options.external_embedding

        parser = ArcHybridLSTM(words, pos, rels, w2i, stored_opt)
        parser.Load(options.model)
        conllu = (os.path.splitext(options.conll_test.lower())[1] == '.conllu')
        tespath = os.path.join(options.output, 'test_pred.conll' if not conllu else 'test_pred.conllu')
        ts = time.time()
        pred = list(parser.Predict(options.conll_test))
        te = time.time()
        transition_utils.write_conll(tespath, pred)

        if not conllu:
            os.system('perl utils/eval.pl -g ' + options.conll_test + ' -s ' + tespath  + ' > ' + tespath + '.txt')
        else:
            os.system('python utils/evaluation_script/conll17_ud_eval.py -v -w utils/evaluation_script/weights.clas ' + options.conll_test + ' ' + tespath + ' > ' + tespath + '.txt')

        print('Finished predicting test', te - ts)

