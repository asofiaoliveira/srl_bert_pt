import xml.etree.ElementTree as ET
from typing import DefaultDict, List, Optional, Iterator, Set, Tuple
import os
from allennlp_models.structured_prediction.models.srl import convert_bio_tags_to_conll_format

# This file reads docs in the XML format and converts them
# to the same format as PropBank.Br.v1.1.conll 
# but only with some columns filled out
# (The rest have "0" in all rows)
# This allows us to use the same dataset reader in all experiments


def role_conversion(role):
    """
    This function simply passes the arguments to the CoNLL format, e.g.:
    arg0 -> A0
    argm-tmp -> AM-TMP
    """
    if ".v" in role:
        role = role.split('.v')[0]
    if role == "argm-pnc":
        # In v1.1, the role argm-pnc is the same as the v2's argm-prp, 
        # so convert to the latter to be uniform
        role = "argm-prp"
    return role[0].upper() + role[3:].upper()

class write:
    def dataset_writer(self, file_path, flags):
        tree = ET.parse(file_path)
        root = tree.getroot()
        body = root.find("body")
        sentences = body.findall("s")
        self.k=0 #count of excluded propositions
        for sentence in sentences:
            s = self.sentence_processing(sentence, flags)
            if s is not None:
                empt = ["0"] * len(s[0]) #empty column
                lines = zip(empt, s[0], empt, empt, empt, empt, empt, empt, s[2], s[3], s[1])
                # I only write in the file the columns that will be needed
                conll_file.writelines('\n'.join("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % l for l in lines))
                conll_file.write("\n\n")
            else: self.k+=1
        print("Number of excluded propositions in %s: %i" % (file_path, self.k))

    def find_arg_words(self, fenode_id, ids, nt, words_with_role): 
        if fenode_id in ids:
            ind = ids.index(fenode_id)
            words_with_role.append(ind)
        elif fenode_id in nt.keys():
            for fenode_id in nt[fenode_id]:
                self.find_arg_words(fenode_id, ids, nt, words_with_role)
        else:
            raise Exception()

    def assign_roles(self, words_with_role, tags, role, words):
        #I do IOB and then convert because of words with 2 roles or roles that appear twice
        tags[words_with_role[0]] = "B-" + role
        last_w = words_with_role[0]
        for list_ind, w_ind in enumerate(words_with_role[1:], 1):
            if w_ind - last_w > 1:
                self.assign_roles(words_with_role[list_ind:], tags, "C-" + role if "C-" not in role else role, words)
                break
            else:
                tags[w_ind] = "I-" + role
                last_w = w_ind
        
    def sentence_processing(self, sentence, flags: bool) -> List[List[str]]:
        ids = []
        words = []
        nt = dict()
        for word in sentence.iter("t"):
            words.append(word.attrib["word"])
            ids.append(word.attrib["id"])

        tags = ["O"] * len(words)
        tags_list = []
        sense = ["-"] * len(words)
        lemma = ["-"] * len(words)

        for node in sentence.iter("nt"):
            i = node.get("id")
            edges = []
            for edge in node:
                edges.append(edge.get("idref"))
            nt[i] = edges
        
        if sentence[2].find("globals").find("global") is not None:
            if sentence[2].find("globals").find("global").get("type") == "WRONGSUBCORPUS":
            # remove propositions with WRONGSUBCORPUS flag
                return None
            #For v2 and buscape, remove LATER, REEXAMINE flags
            if flags:
                if sentence[2].find("globals").find("global").get("type") in ["REEXAMINE", "LATER"]:
                    return None
        
        for fe in sentence[2].iter("fe"):
            role = fe.attrib["name"] 
            if role not in ["referente", "argm-med", "argm-pin"]:
                words_with_role = []
                words_wo_role = []
                #Collect which words are going to have the role
                for fenode in fe:
                    fenode_id = fenode.get("idref")
                    self.find_arg_words(fenode_id, ids, nt, words_with_role)
                    words_with_role.sort()
                
                words_wo_role += [x for x in range(words_with_role[0],words_with_role[-1]) \
                    if x not in words_with_role]

                if role in tags_list and role[3] != 'm':
                    # Removing propositions that have more than one core role (12 in total)
                    return None
                if any([tags[w] != "O" for w in words_with_role]):
                    # Remove propositions that have more than one role in a word
                    return None
                
                tags_list.append(role)
                role = role_conversion(role)
                self.assign_roles(words_with_role, tags, role, words)

        for s in sentence[2].iter("wordtag"):
            if s.attrib["name"] == "sentido":
                verb_id = s.attrib["idref"]
                if tags[ids.index(verb_id)] != "O":
                    #verb index annotation error
                    return None
                tags[ids.index(verb_id)] = "B-V"
                sense[ids.index(verb_id)] = str(s.text.split('.')[1])
                lemma[ids.index(verb_id)] = s.text.split('.')[0]
                self.verb = s.text.split('.')[0]

        if tags.count("B-V") == 0:
            #if there are no verbs (annotation error) don't append sentence
            return None

        return [words, convert_bio_tags_to_conll_format(tags), sense, lemma]

if __name__ == "__main__":
    #make directory conll_data
    if not os.path.exists("./data/conll_data"):
        os.mkdir("./data/conll_data")

    # Create CoNLL format files for PropBank.Br CV (v1.1 + v2) and for Buscapé
    conll_file = open("./data/conll_data/train.conll", "w", encoding="UTF-8")
    write().dataset_writer("./data/xml_data/PropBankBr_v1.1.xml", False)

    conll_file = open("./data/conll_data/train.conll", "a", encoding="UTF-8")
    write().dataset_writer("./data/xml_data/PropBank.Br v.2.xml", True)

    conll_file = open("./data/conll_data/buscape.conll", "w", encoding="UTF-8")
    write().dataset_writer("./data/xml_data/Buscapé.xml", True)
