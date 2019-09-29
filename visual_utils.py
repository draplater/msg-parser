import os

import requests
from lxml import etree

from graph_utils import Graph


def to_wordlist_linguaview_xml(sentence):
    """
    Get wordlist in LinguaView XML representation.
    :return:
    """
    word_list = etree.Element("wordlist")
    word_list.attrib["length"] = str(len(sentence))
    for i in sentence:
        tok = etree.Element("tok")
        tok.attrib["id"] = str(i.id)
        tok.attrib["head"] = i.form
        word_list.append(tok)
    return word_list


def to_deepdep_linguaview_xml(sentence):
    deepdep = etree.Element("deepdep")
    deepdep.text = "\n".join("({}, {}, {})".format(s, t, r) for s, r, t in sentence.generate_edges())
    return deepdep


def get_sentence_node(sentence, sentence_id=1):
    """
    convert Linguistic Data to "sentence" field in LinguaView XML
    :param const_tree: CFG Tree
    :param root: LFG Tree
    :return: <sentence> ... </sentence>
    """

    sentence_node = etree.Element("sentence")
    sentence_node.attrib["id"] = str(sentence_id)
    sentence_node.append(to_wordlist_linguaview_xml(sentence)) # wordlist
    sentence_node.append(to_deepdep_linguaview_xml(sentence)) # constree

    return sentence_node


def get_linguaview_node(sentence_node):
    """
    get "viewer" node of LinguaView XML
    :param sentence_node: one or many linguaview sentences
    :return:
    """
    root = etree.Element("viewer")
    if isinstance(sentence_node, list):
        for idx, i in enumerate(sentence_node):
            i.attrib["id"] = str(idx + 1)
            root.append(i)
    else:
        # noinspection PyUnresolvedReferences
        sentence_node.attrib["id"] = str(1)
        root.append(sentence_node)
    return root


def graph_to_xml(graph):
    return get_linguaview_node(get_sentence_node(graph))


def dep_xml_to_svg(dep_xml):
    output_string = etree.tostring(dep_xml, pretty_print=True, xml_declaration=True,
                                   encoding="UTF-8")
    #print(output_string.decode("utf-8"))
    r = requests.post("http://172.31.222.35:8000/api/dep_from_xml",
                      headers={"Content-Type": "application/xml"},
                      data=output_string)
    r.encoding = "utf-8"
    return r.content


def graph_to_svg(dep):
    return dep_xml_to_svg(graph_to_xml(dep))


def main():
    graphs = Graph.from_file(os.path.expanduser("~/Development/large-data/grbank/dev_output.sdp"))
    svg = graph_to_svg(graphs[0])
    with open("/tmp/b.svg", "w") as f:
        f.write(svg)


if __name__ == '__main__':
    main()