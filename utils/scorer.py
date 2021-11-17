import torch
import json
from utils.constant import BIO_TO_ID

def is_normal_triple(triples, is_relation_first=False):
    """
    normal triples means triples are not over lap in entity.
    example [e1,e2,r1, e3,e4,r2]
    :param triples
    :param is_relation_first
    :return:

    >>> is_normal_triple([1,2,3, 4,5,0])
    True
    >>> is_normal_triple([1,2,3, 4,5,3])
    True
    >>> is_normal_triple([1,2,3, 2,5,0])
    False
    >>> is_normal_triple([1,2,3, 1,2,0])
    False
    >>> is_normal_triple([1,2,3, 4,5,0], is_relation_first=True)
    True
    >>> is_normal_triple([1,2,3, 4,5,3], is_relation_first=True)
    False
    >>> is_normal_triple([1,2,3, 2,5,0], is_relation_first=True)
    True
    >>> is_normal_triple([1,2,3, 1,2,0], is_relation_first=True)
    False
    """
    entities = set()
    for i, e in enumerate(triples):
        key = 0 if is_relation_first else 2
        if i % 3 != key:
            entities.add(e)
    return len(entities) == 2 * len(triples) / 3


def is_multi_label(triples, is_relation_first=False):
    """
    :param triples:
    :param is_relation_first:
    :return:
    >>> is_multi_label([1,2,3, 4,5,0])
    False
    >>> is_multi_label([1,2,3, 4,5,3])
    False
    >>> is_multi_label([1,2,3, 2,5,0])
    False
    >>> is_multi_label([1,2,3, 1,2,0])
    True
    >>> is_multi_label([1,2,3, 4,5,0], is_relation_first=True)
    False
    >>> is_multi_label([1,2,3, 4,5,3], is_relation_first=True)
    False
    >>> is_multi_label([1,5,0, 2,5,0], is_relation_first=True)
    True
    >>> is_multi_label([1,2,3, 1,2,0], is_relation_first=True)
    False
    """
    if is_normal_triple(triples, is_relation_first):
        return False
    if is_relation_first:
        entity_pair = [tuple(triples[3 * i + 1: 3 * i + 3]) for i in range(len(triples) // 3)]
    else:
        entity_pair = [tuple(triples[3 * i: 3 * i + 2]) for i in range(len(triples) // 3)]
    # if is multi label, then, at least one entity pair appeared more than once
    return len(entity_pair) != len(set(entity_pair))


def is_over_lapping(triples, is_relation_first=False):
    """
    :param triples:
    :param is_relation_first:
    :return:
    >>> is_over_lapping([1,2,3, 4,5,0])
    False
    >>> is_over_lapping([1,2,3, 4,5,3])
    False
    >>> is_over_lapping([1,2,3, 2,5,0])
    True
    >>> is_over_lapping([1,2,3, 1,2,0])
    False
    >>> is_over_lapping([1,2,3, 4,5,0], is_relation_first=True)
    False
    >>> is_over_lapping([1,2,3, 4,5,3], is_relation_first=True)
    True
    >>> is_over_lapping([1,5,0, 2,5,0], is_relation_first=True)
    False
    >>> is_over_lapping([1,2,3, 1,2,0], is_relation_first=True)
    True
    """
    if is_normal_triple(triples, is_relation_first):
        return False
    if is_relation_first:
        entity_pair = [tuple(triples[3 * i + 1: 3 * i + 3]) for i in range(len(triples) // 3)]
    else:
        entity_pair = [tuple(triples[3 * i: 3 * i + 2]) for i in range(len(triples) // 3)]
    # remove the same entity_pair, then, if one entity appear more than once, it's overlapping
    entity_pair = set(entity_pair)
    entities = []
    for pair in entity_pair:
        entities.extend(pair)
    entities = set(entities)
    return len(entities) != 2 * len(entity_pair)


# read data
def read_json(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            a_data = json.loads(line)
            data.append(a_data)
    return data

def sta(labels_RC, labels_NER, logits_RC, logits_NER):
    labels_num, logits_num, right_num = 0, 0, 0
    # total num of golden relations
    labels_num = labels_RC.sum().item()
    # total num of predicted relations
    logits_RC = torch.where(logits_RC>=0.5, torch.ones_like(logits_RC), torch.zeros_like(logits_RC))
    logits_num = logits_RC.sum().item()
    # total num of predicted right relations
    # right relations
    right_RC = logits_RC.cuda()+labels_RC.cuda()
    right_RC = torch.where(right_RC==2, torch.ones_like(right_RC), torch.zeros_like(right_RC))
    # right entities
    right_EN2RC_mask = get_right_entity_pair(labels_NER, logits_NER)
    right_RC = right_RC * right_EN2RC_mask
    # right num of predicted relations
    right_num = right_RC.sum().item()
    return labels_num, logits_num, right_num

def get_right_entity_pair(labels_NER, logits_NER):
    NER_argmax = torch.argmax(logits_NER, dim=2)
    rp_list = []
    for i in range(NER_argmax.size(0)):
        rp_list.append(find_right(labels_NER[i], NER_argmax[i]))
    assert(len(rp_list) == NER_argmax.size(0))
    ret = torch.zeros((logits_NER.size(0), logits_NER.size(1), logits_NER.size(1), 1))
    for i in range(len(rp_list)):
        if len(rp_list[i]) <= 1:
            continue
        epairs = get_pairs(rp_list[i])
        for ep in epairs:
            ret[i][ep[0]][ep[1]][0] = 1.
            ret[i][ep[1]][ep[0]][0] = 1.
    return ret.cuda()

def find_right(label_NER, logit_NER):
    stack, ner_right = [], []
    # to list
    label_NER = label_NER.cpu().detach().numpy().tolist()
    logit_NER = logit_NER.cpu().detach().numpy().tolist()
    for i, v in enumerate(label_NER):
        if v == BIO_TO_ID['S'] and label_NER[i] == logit_NER[i]:
            ner_right.append(i)
        elif v == BIO_TO_ID['B']:
            stack.append(i)
        elif v == BIO_TO_ID['E']:
            start = stack.pop(0)
            end = i+1
            if label_NER[start:end] == logit_NER[start:end]:
                ner_right.append(i)
    return ner_right

def get_pairs(ens):
    ret = []
    for i in range(len(ens)):
        for j in range(i+1,len(ens)):
            ret.append((ens[i], ens[j]))
    return ret
        

def analysis_data(filename):
    data = read_json(filename)
    print(len(data))
    normal_count, multi_label_count, over_lapping_count = 0, 0, 0
    for d in data:
        sent_triples = []
        for triple in d['relationMentions']:
            sent_triples.extend([triple['em1Text'][0], triple['em2Text'][0], triple['label']])
        normal_count += 1 if is_normal_triple(sent_triples) else 0
        multi_label_count += 1 if is_multi_label(sent_triples) else 0
        over_lapping_count += 1 if is_over_lapping(sent_triples) else 0
    print('Normal: {}, EPO: {}, SEO: {}'.format(normal_count, multi_label_count, over_lapping_count))


# from RC label to analysis_data
'''
from utils.scorer import is_normal_triple, is_multi_label, is_over_lapping
def get_triple_list(instance):
    normal, epo, seo = 0, 0, 0
    tmp = instance.sum(dim=2)
    tmp1 = torch.where(tmp>=2, torch.ones_like(tmp), torch.zeros_like(tmp))
    if tmp1.sum() > 0:
        epo = 1
    tmp2 = torch.where(tmp!=0, torch.ones_like(tmp), torch.zeros_like(tmp))
    tmp3 = tmp2.sum(dim=0)
    tmp2 = tmp2.sum(dim=1)
    tmp22 = torch.where(tmp2 != 0, torch.ones_like(tmp2), torch.zeros_like(tmp2))
    tmp33 = torch.where(tmp3 != 0, torch.ones_like(tmp3), torch.zeros_like(tmp3))
    tmp2 = torch.where(tmp2 >1, torch.ones_like(tmp2), torch.zeros_like(tmp2))
    tmp3 = torch.where(tmp3 >1, torch.ones_like(tmp3), torch.zeros_like(tmp3))
    if tmp2.sum() > 0:
        seo = 1
    if tmp3.sum() > 0:
        seo = 1
    if (tmp22*tmp33).sum() > 0:
        seo = 1
    if epo != 1 and seo != 1:
        normal = 1
    return normal, epo, seo
normal_count, multi_label_count, over_lapping_count = 0, 0, 0
c = 0
for batch in train_batch:
    # 0: tokens, 1: pos, 2: mask_s, 3: RC_labels 4: NER_labels
    RC_labels = batch[3]
    for i in range(RC_labels.size(0)):
        normal, epo, seo = get_triple_list(RC_labels[i])
        normal_count += normal
        multi_label_count += epo
        over_lapping_count += seo
    c += 1
    print(c)
print('Normal: {}, EPO: {}, SEO: {}'.format(normal_count, multi_label_count, over_lapping_count))
'''

