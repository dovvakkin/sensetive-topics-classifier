import json
import torch

import numpy as np
import streamlit as st

from annotated_text import annotated_text
from transformers_interpret import SequenceClassificationExplainer
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_NAME = 'Skoltech/russian-sensitive-topics'

@st.cache(allow_output_mutation=True)
def init_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME);

    cls_explainer = SequenceClassificationExplainer(
        model,
        tokenizer)

    return tokenizer, model, cls_explainer


@st.cache(allow_output_mutation=True)
def init_target_mapping():
    with open("id2topic.json") as f:
        target_vaiables_id2topic_dict = json.load(f)

    return target_vaiables_id2topic_dict


def adjust_multilabel(y, target_vaiables_id2topic_dict, is_pred = False):
    y_adjusted = []
    for y_c in y:
        y_test_curr = [0]*19
        index = str(int(np.argmax(y_c)))
        y_c = target_vaiables_id2topic_dict[index]
    return y_c


def get_prediction_and_explaination(sent, tokenizer, model, cls_explainer, target_vaiables_id2topic_dict):
    tokenized = tokenizer.batch_encode_plus(
        [sent],
        max_length = 512,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False)

    tokens_ids, mask = torch.tensor(tokenized['input_ids']),\
                       torch.tensor(tokenized['attention_mask'])

    with torch.no_grad():
        model_output = model(tokens_ids,mask)

    word_attributions = cls_explainer(sent)

    return adjust_multilabel(model_output['logits'],
                             target_vaiables_id2topic_dict,
                             is_pred = True),
           word_attributions


def main():
    tokenizer, model, cls_explainer = init_model()
    target_vaiables_id2topic_dict = init_target_mapping()

    st.title('Sensitive topics detection')
    with st.form('Form1'):
        input_text = st.text_input('Введите текст', value='Сегодня хороший день')
        #method = st.selectbox(label='Выберите классификатор',
        #                      options=['TFIDF + Logistic Regression', 'Fine-tuned BERT'])
        #st.caption('Выберите классификатор:')

        submit_button = st.form_submit_button(label='Find sensititve topics')

    if submit_button:
        st.caption('Result:')

        topic, word_attrs = get_prediction_and_explaination(input_text,
                                                            tokenizer,
                                                            model,
                                                            cls_explainer,
                                                            target_vaiables_id2topic_dict)
        st.write(topic)


if __name__ == '__main__':
    main()

# annotated_text(
#     "This ",
#     ("is", "verb", "#8ef"),
#     " some ",
#     ("annotated", "adj", "#faa"),
#     ("text", "noun", "#afa"),
#     " for those of ",
#     ("you", "pronoun", "#fea"),
#     " who ",
#     ("like", "verb", "#8ef"),
#     " this sort of ",
#     ("thing", "noun", "#afa"),
#     "."
# )
