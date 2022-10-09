import kfp

from typing import NamedTuple

from kfp.v2.dsl import pipeline
from kfp.v2.dsl import component
from kfp.v2 import compiler

if __name__ == '__main__':
    data = json.load(open('config.json'))
    f = open("demo.txt", "w")
    f.write(str(data['key3']))
    f.write(str(data['key2']))
    f.write(str(data['key1']))
    f.close()