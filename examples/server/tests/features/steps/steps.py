import socket
import threading
from contextlib import closing

import openai
import requests
from behave import step


@step(
    u"a server listening on {server_fqdn}:{server_port} with {n_slots} slots, {seed} as seed and {api_key} as api key")
def step_server_config(context, server_fqdn, server_port, n_slots, seed, api_key):
    context.server_fqdn = server_fqdn
    context.server_port = int(server_port)
    context.n_slots = int(n_slots)
    context.seed = int(seed)
    context.base_url = f'http://{context.server_fqdn}:{context.server_port}'

    context.completions = []
    context.completion_threads = []
    context.prompts = []

    context.api_key = api_key
    openai.api_key = context.api_key


@step(u"the server is {expecting_status}")
def step_wait_for_the_server_to_be_started(context, expecting_status):
    match expecting_status:
        case 'starting':
            server_started = False
            while not server_started:
                with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                    result = sock.connect_ex((context.server_fqdn, context.server_port))
                    if result == 0:
                        return 0
        case 'loading model':
            wait_for_health_status(context, 503, 'loading model')
        case 'healthy':
            wait_for_health_status(context, 200, 'ok')
        case 'ready' | 'idle':
            wait_for_health_status(context, 200, 'ok',
                                   params={'fail_on_no_slot': True},
                                   slots_idle=context.n_slots,
                                   slots_processing=0)
            request_slots_status(context, [
                {'id': 0, 'state': 0},
                {'id': 1, 'state': 0}
            ])
        case 'busy':
            wait_for_health_status(context, 503, 'no slot available',
                                   params={'fail_on_no_slot': True},
                                   slots_idle=0,
                                   slots_processing=context.n_slots)
            request_slots_status(context, [
                {'id': 0, 'state': 1},
                {'id': 1, 'state': 1}
            ])
        case _:
            assert False, "unknown status"


@step(u'all slots are {expected_slot_status_string}')
def step_all_slots_status(context, expected_slot_status_string):
    match expected_slot_status_string:
        case 'idle':
            expected_slot_status = 0
        case 'busy':
            expected_slot_status = 1
        case _:
            assert False, "unknown status"

    expected_slots = []
    for slot_id in range(context.n_slots):
        expected_slots.append({
            'id': slot_id,
            'state': expected_slot_status
        })
    request_slots_status(context, expected_slots)


@step(u'a completion request')
def step_request_completion(context):
    request_completion(context, context.prompts.pop(), context.n_predict, context.user_api_key)
    context.user_api_key = None


@step(u'{predicted_n} tokens are predicted')
def step_n_tokens_predicted(context, predicted_n):
    if int(predicted_n) > 0:
        assert_n_tokens_predicted(context.completions[0], int(predicted_n))


@step(u'a user prompt {user_prompt}')
def step_user_prompt(context, user_prompt):
    context.user_prompt = user_prompt


@step(u'a system prompt {system_prompt}')
def step_system_prompt(context, system_prompt):
    context.system_prompt = system_prompt


@step(u'a model {model}')
def step_model(context, model):
    context.model = model


@step(u'{max_tokens} max tokens to predict')
def step_max_tokens(context, max_tokens):
    context.n_predict = int(max_tokens)


@step(u'streaming is {enable_streaming}')
def step_streaming(context, enable_streaming):
    context.enable_streaming = enable_streaming == 'enabled' or bool(enable_streaming)


@step(u'a user api key {user_api_key}')
def step_user_api_key(context, user_api_key):
    context.user_api_key = user_api_key


@step(u'a user api key ')
def step_user_api_key(context):
    context.user_api_key = None


@step(u'an OAI compatible chat completions request with an api error {api_error}')
def step_oai_chat_completions(context, api_error):
    oai_chat_completions(context, context.user_prompt, api_error=api_error == 'raised')
    context.user_api_key = None


@step(u'a prompt')
def step_a_prompt(context):
    context.prompts.append(context.text)


@step(u'a prompt {prompt}')
def step_a_prompt_prompt(context, prompt):
    context.prompts.append(prompt)


@step(u'concurrent completion requests')
def step_concurrent_completion_requests(context):
    concurrent_requests(context, request_completion, context.n_predict, context.user_api_key)


@step(u'concurrent OAI completions requests')
def step_oai_chat_completions(context):
    concurrent_requests(context, oai_chat_completions, context.user_api_key)


@step(u'all prompts are predicted')
def step_all_prompts_are_predicted(context):
    for completion_thread in context.completion_threads:
        completion_thread.join()
    assert len(context.completions) == len(context.completion_threads)
    for completion in context.completions:
        assert_n_tokens_predicted(completion)


@step(u'embeddings are computed for')
def step_compute_embedding(context):
    response = requests.post(f'{context.base_url}/embedding', json={
        "content": context.text,
    })
    assert response.status_code == 200
    context.embeddings = response.json()['embedding']


@step(u'embeddings are generated')
def step_compute_embeddings(context):
    assert len(context.embeddings) > 0


@step(u'an OAI compatible embeddings computation request for')
def step_oai_compute_embedding(context):
    openai.api_base = f'{context.base_url}/v1'
    embeddings = openai.Embedding.create(
        model=context.model,
        input=context.text,
    )
    context.embeddings = embeddings


@step(u'tokenizing')
def step_tokenize(context):
    context.tokenized_text = context.text
    response = requests.post(f'{context.base_url}/tokenize', json={
        "content": context.tokenized_text,
    })
    assert response.status_code == 200
    context.tokens = response.json()['tokens']


@step(u'tokens can be detokenize')
def step_detokenize(context):
    assert len(context.tokens) > 0
    response = requests.post(f'{context.base_url}/detokenize', json={
        "tokens": context.tokens,
    })
    assert response.status_code == 200
    # FIXME the detokenize answer contains a space prefix ? see #3287
    assert context.tokenized_text == response.json()['content'].strip()


@step(u'an OPTIONS request is sent from {origin}')
def step_options_request(context, origin):
    options_response = requests.options(f'{context.base_url}/v1/chat/completions',
                                        headers={"Origin": origin})
    assert options_response.status_code == 200
    context.options_response = options_response


@step(u'CORS header {cors_header} is set to {cors_header_value}')
def step_check_options_header_value(context, cors_header, cors_header_value):
    assert context.options_response.headers[cors_header] == cors_header_value


def concurrent_requests(context, f_completion, *argv):
    context.completions.clear()
    context.completion_threads.clear()
    for prompt in context.prompts:
        completion_thread = threading.Thread(target=f_completion, args=(context, prompt, *argv))
        completion_thread.start()
        context.completion_threads.append(completion_thread)
    context.prompts.clear()


def request_completion(context, prompt, n_predict=None, user_api_key=None):
    origin = "my.super.domain"
    headers = {
        'Origin': origin
    }
    if 'user_api_key' in context:
        headers['Authorization'] = f'Bearer {user_api_key}'

    response = requests.post(f'{context.base_url}/completion',
                             json={
                                 "prompt": prompt,
                                 "n_predict": int(n_predict) if n_predict is not None else context.n_predict,
                                 "seed": context.seed
                             },
                             headers=headers)
    if n_predict is not None and n_predict > 0:
        assert response.status_code == 200
        assert response.headers['Access-Control-Allow-Origin'] == origin
        context.completions.append(response.json())
    else:
        assert response.status_code == 401


def oai_chat_completions(context, user_prompt, api_error=None):
    openai.api_key = context.user_api_key
    openai.api_base = f'{context.base_url}/v1/chat'
    try:
        chat_completion = openai.Completion.create(
            messages=[
                {
                    "role": "system",
                    "content": context.system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            model=context.model,
            max_tokens=context.n_predict,
            stream=context.enable_streaming,
            seed=context.seed
        )
    except openai.error.APIError:
        if api_error:
            openai.api_key = context.api_key
            return
    openai.api_key = context.api_key
    if context.enable_streaming:
        completion_response = {
            'content': '',
            'timings': {
                'predicted_n': 0
            }
        }
        for chunk in chat_completion:
            assert len(chunk.choices) == 1
            delta = chunk.choices[0].delta
            if 'content' in delta:
                completion_response['content'] += delta['content']
                completion_response['timings']['predicted_n'] += 1
        context.completions.append(completion_response)
    else:
        assert len(chat_completion.choices) == 1
        context.completions.append({
            'content': chat_completion.choices[0].message,
            'timings': {
                'predicted_n': chat_completion.usage.completion_tokens
            }
        })


def assert_n_tokens_predicted(completion_response, expected_predicted_n=None):
    content = completion_response['content']
    n_predicted = completion_response['timings']['predicted_n']
    assert len(content) > 0, "no token predicted"
    if expected_predicted_n is not None:
        assert n_predicted == expected_predicted_n, (f'invalid number of tokens predicted:'
                                                     f' "{n_predicted}" <> "{expected_predicted_n}"')


def wait_for_health_status(context, expected_http_status_code,
                           expected_health_status,
                           params=None,
                           slots_idle=None,
                           slots_processing=None):
    while True:
        health_response = requests.get(f'{context.base_url}/health', params)
        status_code = health_response.status_code
        health = health_response.json()
        if (status_code == expected_http_status_code
                and health['status'] == expected_health_status
                and (slots_idle is None or health['slots_idle'] == slots_idle)
                and (slots_processing is None or health['slots_processing'] == slots_processing)):
            break


def request_slots_status(context, expected_slots):
    slots_response = requests.get(f'{context.base_url}/slots')
    assert slots_response.status_code == 200
    slots = slots_response.json()
    assert len(slots) == len(expected_slots)
    for expected, slot in zip(expected_slots, slots):
        for key in expected:
            assert expected[key] == slot[key], f"expected[{key}] != slot[{key}]"
