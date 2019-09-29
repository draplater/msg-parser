#!/usr/bin/python
# encoding: utf-8

from __future__ import unicode_literals

import codecs
import json

import requests
from jsonrpc_requests import Server


def main():
    api = Server("http://127.0.0.1:9996/api")
    with codecs.open("web/src/defaultSentence.txt", encoding="utf-8") as f:
        sentence = f.read().strip()

    with codecs.open("web/src/defaultTree.txt", "w", encoding="utf-8") as f:
        f.write(api.phrase_parse(sentence))

    with codecs.open("web/src/defaultEDS.txt", "w", encoding="utf-8") as f:
        f.write(api.hrg_parse(sentence))

if __name__ == "__main__":
    main()
