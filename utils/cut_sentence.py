import re


def cut_sentences_v1(sent):
    """
    the first rank of sentence cut
    """
    sent = re.sub('([。！？\?])([^”’])', r"\1\n\2", sent)  # single-character sentence terminators
    sent = re.sub('(\.{6})([^”’])', r"\1\n\2", sent)  # English ellipsis
    sent = re.sub('(\…{2})([^”’])', r"\1\n\2", sent)  # Chinese ellipsis
    sent = re.sub('([。！？\?][”’])([^，。！？\?])', r"\1\n\2", sent)
    # If a terminator precedes the quotation mark, the quotation mark is the actual
    # sentence end, so the sentence break (\n) should go after the closing quote.
    return sent.split("\n")


def cut_sentences_v2(sent):
    """
    the second rank of spilt sentence, split '；' | ';'
    """
    sent = re.sub('([；;])([^”’])', r"\1\n\2", sent)
    return sent.split("\n")


def cut_sent_for_bert(text, max_seq_len):
    # Split into sentences, do a fine-grained split first, then merge short ones
    sentences = []

    # Fine-grained split
    sentences_v1 = cut_sentences_v1(text)
    # print("sentences_v1=", sentences_v1)
    for sent_v1 in sentences_v1:
        if len(sent_v1) > max_seq_len - 2:
            sentences_v2 = cut_sentences_v2(sent_v1)
            sentences.extend(sentences_v2)
        else:
            sentences.append(sent_v1)

    assert ''.join(sentences) == text

    # Merge
    merged_sentences = []
    start_index_ = 0

    while start_index_ < len(sentences):
        tmp_text = sentences[start_index_]

        end_index_ = start_index_ + 1
        # For the BERT model, note that the max length here needs to subtract 2 (for [CLS] and [SEP])
        while end_index_ < len(sentences) and \
                len(tmp_text) + len(sentences[end_index_]) <= max_seq_len - 2:
            tmp_text += sentences[end_index_]
            end_index_ += 1

        start_index_ = end_index_

        merged_sentences.append(tmp_text)

    return merged_sentences


def refactor_labels(sent, labels, start_index):
    """
    After splitting sentences, refactor the label offsets.
    :param sent: the re-merged sentence after splitting
    :param labels: the document-level labels
    :param start_index: the offset of this sentence within the document
    :return (type, entity, offset)
    """
    new_labels = []
    end_index = start_index + len(sent)
    # _label: (T_id, entity type, entity start position, entity end position, entity text)
    for _label in labels:
        if start_index <= _label[2] <= _label[3] <= end_index:
            new_offset = _label[2] - start_index

            assert sent[new_offset: new_offset + len(_label[-1])] == _label[-1]

            new_labels.append((_label[1], _label[-1], new_offset))
        # Case where the label is truncated by the split
        elif _label[2] < end_index < _label[3]:
            raise RuntimeError(f'{sent}, {_label}')

    return new_labels


if __name__ == '__main__':
    raw_examples = [{
        "text": "深圳市沙头角保税区今后五年将充分发挥保税区的区位优势和政策优势，以高新技术产业为先导，积极调整产品结构，实施以转口贸易和仓储业为辅助的经营战略。把沙头角保税区建成按国际惯例运作、国内领先的特殊综合经济区域，使其成为该市外向型经济的快速增长点。",
        "labels": [
            [
                "T0",
                "GPE",
                0,
                3,
                "深圳市"
            ],
            [
                "T1",
                "GPE",
                3,
                6,
                "沙头角"
            ],
            [
                "T2",
                "LOC",
                6,
                9,
                "保税区"
            ],
            [
                "T3",
                "LOC",
                18,
                21,
                "保税区"
            ],
            [
                "T4",
                "GPE",
                73,
                76,
                "沙头角"
            ],
            [
                "T5",
                "LOC",
                76,
                79,
                "保税区"
            ]
        ]
    }]
    for i, item in enumerate(raw_examples):
        text = item['text']
        print(text[:90])
        sentences = cut_sent_for_bert(text, 90)
        start_index = 0

        for sent in sentences:
            labels = refactor_labels(sent, item['labels'], start_index)
            start_index += len(sent)

            print(sent)
            print(labels)