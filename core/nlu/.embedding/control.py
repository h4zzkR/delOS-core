import sys, os; sys.path.insert(0,'../delos')
import argparse
import time, datetime
import os
from termcolor import colored
from settings import ROOT_DIR
from core.utils.logger import init_logger
from core.nlu.embedder.main import EmbeddingLoader, EmbeddingTrainer


parser = argparse.ArgumentParser(description='Класс для управления embeddings')

parser.add_argument(
    '--mode',
    type=str,
    default='load',
    help='load/train Загрузить готовые, либо обучить на корпусе текстов.'
)

parser.add_argument(
    '--model',
    type=str,
    default=None,
    help='On load: Относительный путь к gensim модели (папка с файлами модели)'
)

parser.add_argument(
    '-et', '--embedding_type',
    type=str,
    default='gensim',
    help='gensim/bert - глобальный тип тренировки embedding. По умолчанию gensim.'
)

parser.add_argument(
    '-gt', '--gensim_type',
    type=str,
    default='fasttext',
    help='w2v/fasttext - тип embeddings для gensim.'
)

parser.add_argument(
    '-t', '--train_dir',
    type=str,
    default=None,
    help='Путь к папке с корпусами для обучения (папки с txt, либо просто txt)'
)

parser.add_argument(
    '-ws', '--window_size',
    type=int,
    default=6,
    help='Размер окна для BOW/CBOW в gensim'
)
parser.add_argument(
    '--emb_dim',
    type=int,
    default=256,
    help='Размерность выходных векторов'
)
parser.add_argument(
    '-m', '--min_count',
    type=int,
    default=10,
    help='Минимальное количество вхождений слова в документе, чтобы gensim добавил его в словарь'
)

parser.add_argument(
    '-e', '--epochs',
    type=int,
    default=15,
    help='Количество итераций обучения.'
)

parser.add_argument(
    '--workers',
    type=int,
    default=2,
    help='Num of workers'
)
parser.add_argument(
    '--key_vectors',
    action='store_true',
    help='Позволяет загрузить/сохранить только keyed vectors без загрузки/сохранения полной модели.'
)

parser.add_argument(
    '-i', '--interactive',
    action='store_true',
    help='Инициализировать и возратить объект'
)

parser.add_argument(
    '--extract_vectors',
    action='store_true',
    default=True,
    help='Помимо сохранения модели сохранить векторы слов в bin/kv'
)

namespace = parser.parse_args()


def contoller():
    mode = namespace.mode
    embedding_type = namespace.embedding_type

    if embedding_type.lower() == 'gensim':
        gensim_type = namespace.gensim_type
        if namespace.train_dir == None:
            namespace.train_dir = os.path.join(ROOT_DIR, 'data/datasets/cleared/')
    else:
        pass

    if mode.lower() == 'train':
        embed = EmbeddingTrainer(
            train_type=namespace.embedding_type, gensim_type=namespace.gensim_type,
            train_dir=namespace.train_dir, window_size=namespace.window_size,
            emb_dim=namespace.emb_dim, min_count=namespace.min_count, epochs=namespace.epochs, 
            num_workers=namespace.workers,
        )
        if namespace.interactive is False:
            embed.init_model()
            embed.train_model()
            embed.save_model(key_vectors=namespace.key_vectors)
            return None
        else:
            return embed
    elif mode.lower() == 'load':
        embed = EmbeddingLoader(
            model=namespace.model, embedding_type=namespace.embedding_type, gensim_type=namespace.gensim_type,
            workers=namespace.workers, key_vectors=namespace.key_vectors
        )
        embed.load()
        return embed




if __name__ == "__main__":
    model = contoller()
    # print('Test:', 'тест -', model.model.wv['тест'])