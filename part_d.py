"""Part D: TF-IDF policy RAG, deterministic tool-using refund agent and resumable runs."""
from pathlib import Path
import json, re, time, sqlite3
from datetime import datetime, timezone
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data'; OUT=ROOT/'outputs'; STATE=ROOT/'state'
OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
for old_checkpoint in STATE.glob('*.json'):
    old_checkpoint.unlink()
POLICY_FILES=['refund_policy.txt','delivery_policy.txt','payment_policy.txt','privacy_policy.txt','escalation_policy.txt']

def load_chunks(overlap=10):
    chunks=[]; step=40-overlap
    for filename in POLICY_FILES:
        words=(DATA/'policies'/filename).read_text(encoding='utf-8').split()
        start=0
        while start < len(words):
            end=min(start+40,len(words))
            chunks.append({'source':filename,'text':' '.join(words[start:end])})
            if end>=len(words): break
            start+=step
    return chunks

CHUNKS=load_chunks(10)
VECTORIZER=TfidfVectorizer()
CHUNK_MATRIX=VECTORIZER.fit_transform([c['text'] for c in CHUNKS])

def retrieve(query,k=3,chunks=None,vectorizer=None,matrix=None):
    chunks=chunks or CHUNKS; vectorizer=vectorizer or VECTORIZER; matrix=matrix if matrix is not None else CHUNK_MATRIX
    q=vectorizer.transform([query]); scores=cosine_similarity(q,matrix).ravel()
    order=scores.argsort()[::-1][:k]
    return [{'source':chunks[i]['source'],'score':float(scores[i]),'text':chunks[i]['text']} for i in order]

def answer(question):
    found=retrieve(question,3)
    context='\n'.join(f"[{item['source']}]\n{item['text']}" for item in found)
    user=f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    response=mock_llm.chat([
        {'role':'system','content':'You are a POLICY ASSISTANT. Answer only from the supplied context. If the context does not contain the answer, say I could not find the information in the policy documents.'},
        {'role':'user','content':user}
    ])
    return response.choices[0].message.content,[x['source'] for x in found]

# Evaluate retrieval and grounding with overlap=10 and overlap=0.
eval_df=pd.read_csv(DATA/'raw'/'policy_eval.csv')
def evaluate_policy(overlap):
    chunks=load_chunks(overlap)
    vec=TfidfVectorizer(); mat=vec.fit_transform([c['text'] for c in chunks])
    answerable_retrieved=answerable_fact=unanswerable_fallback=0; failures=[]
    for _,row in eval_df.iterrows():
        q=row.question
        got=retrieve(q,3,chunks,vec,mat)
        sources=[x['source'] for x in got]
        if row.expected_doc!='NONE':
            if row.expected_doc in sources: answerable_retrieved+=1
            ctx='\n'.join(f"[{x['source']}]\n{x['text']}" for x in got)
            resp=mock_llm.chat([
                {'role':'system','content':'You are a POLICY ASSISTANT. Answer only from the supplied context; say I could not find the information otherwise.'},
                {'role':'user','content':f"CONTEXT:\n{ctx}\nQUESTION:\n{q}"}
            ]).choices[0].message.content
            if str(row.expected_fact) in resp: answerable_fact+=1
            else: failures.append(q)
        else:
            ctx='\n'.join(f"[{x['source']}]\n{x['text']}" for x in got)
            resp=mock_llm.chat([
                {'role':'system','content':'You are a POLICY ASSISTANT. Answer only from the supplied context; say I could not find the information otherwise.'},
                {'role':'user','content':f"CONTEXT:\n{ctx}\nQUESTION:\n{q}"}
            ]).choices[0].message.content
            if resp.startswith('I could not find'): unanswerable_fallback+=1
    return {'chunks':len(chunks),'answerable_retrieval':answerable_retrieved,'answerable_fact':answerable_fact,'unanswerable_fallback':unanswerable_fallback,'failed_questions':failures}

# Tool descriptions and JSON-style parameter specifications.
TOOLS_SPEC=[
 {'name':'get_order','description':'Look up a single order in the SQLite order database.','parameters':{'type':'object','properties':{'order_id':{'type':'string'}},'required':['order_id']}},
 {'name':'search_policy','description':'Retrieve the most relevant policy chunks for a query.','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}},
 {'name':'calculate_refund','description':'Calculate the late-delivery refund under the supplied policy rules.','parameters':{'type':'object','properties':{'order_amount':{'type':'number'},'delivery_minutes':{'type':['integer','null']}},'required':['order_amount','delivery_minutes']}},
 {'name':'issue_refund','description':'Record an automatically approved refund in the simulated ledger.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}},
 {'name':'escalate_to_human','description':'Record a refund that requires human approval.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}}
]

def get_order(order_id):
    with sqlite3.connect(DATA/'instamart.db') as conn:
        row=conn.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',(order_id,)).fetchone()
    if row is None: return {'error':'order not found'}
    return {'order_id':row[0],'status':row[1],'order_amount':float(row[2]),'delivery_minutes':None if row[3] is None else int(row[3])}

def search_policy(query): return {'results':retrieve(query,k=3)}
def calculate_refund(order_amount,delivery_minutes):
    amount=float(order_amount) if delivery_minutes is not None and int(delivery_minutes)>45 else 0.0
    return {'refund_inr':round(amount,2)}

def _ledger(tool_name,order_id,refund_inr):
    entry={'timestamp':datetime.now(timezone.utc).isoformat(),'tool_name':tool_name,'order_id':order_id,'refund_inr':round(float(refund_inr),2)}
    with (OUT/'refund_ledger.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(entry)+'\n')
    return {'refund_inr':round(float(refund_inr),2)}

def issue_refund(order_id,refund_inr): return _ledger('issue_refund',order_id,refund_inr)
def escalate_to_human(order_id,refund_inr): return _ledger('escalate_to_human',order_id,refund_inr)
TOOL_MAP={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}

class SimulatedCrash(RuntimeError): pass

def _save_checkpoint(run_id,scenario_id,messages,tokens_used,llm_calls,steps,executed_tools):
    payload={'run_id':run_id,'scenario_id':scenario_id,'messages':messages,'tokens_used':tokens_used,'llm_calls':llm_calls,
             'step_number':len(steps),'steps':steps,'executed_tools':executed_tools}
    (STATE/f'{run_id}.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')

def _continue_agent(state,max_iterations=6,token_limit=6000,time_limit_seconds=10,crash_after_tool_count=None):
    started=time.monotonic()
    messages=state['messages']; tokens_used=state['tokens_used']; llm_calls=state['llm_calls']; steps=state['steps']; executed=state['executed_tools']
    while llm_calls < max_iterations:
        if time.monotonic()-started >= time_limit_seconds:
            return {'status':'FALLBACK_TIMEOUT','final_answer':'FALLBACK_TIMEOUT: hand off to a human agent; time limit reached.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'executed_tools':executed}
        response=mock_llm.chat(messages,tools=TOOLS_SPEC)
        llm_calls+=1; tokens_used+=int(response.usage.total_tokens)
        if tokens_used>token_limit:
            return {'status':'FALLBACK_TOKEN_LIMIT','final_answer':'FALLBACK_TOKEN_LIMIT: hand off to a human agent; token limit reached.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'executed_tools':executed}
        msg=response.choices[0].message
        tool_calls=msg.tool_calls or []
        if not tool_calls:
            final=msg.content or ''
            status=final.split(':',1)[0] if ':' in final else 'COMPLETED'
            return {'status':status,'final_answer':final,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'executed_tools':executed}
        assistant_calls=[]
        for tc in tool_calls:
            args=tc.arguments if isinstance(tc.arguments,dict) else json.loads(tc.arguments)
            assistant_calls.append({'id':tc.id,'name':tc.name,'arguments':args})
        messages.append({'role':'assistant','content':None,'tool_calls':assistant_calls})
        for tc in tool_calls:
            args=tc.arguments if isinstance(tc.arguments,dict) else json.loads(tc.arguments)
            if tc.name not in TOOL_MAP: observation={'error':'unknown tool'}
            else: observation=TOOL_MAP[tc.name](**args)
            executed.append(tc.name)
            steps.append({'tool_name':tc.name,'arguments':args,'observation':observation})
            messages.append({'role':'tool','tool_call_id':tc.id,'name':tc.name,'content':json.dumps(observation)})
            _save_checkpoint(state['run_id'],state['scenario_id'],messages,tokens_used,llm_calls,steps,executed)
            if crash_after_tool_count is not None and len(executed)==crash_after_tool_count:
                raise SimulatedCrash(f'simulated crash after tool execution {crash_after_tool_count}')
    return {'status':'FALLBACK_MAX_ITERATIONS','final_answer':'FALLBACK_MAX_ITERATIONS: hand off to a human agent; iteration limit reached.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'executed_tools':executed}

def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after_tool_count=None):
    run_id=run_id or f'{scenario_id}_{int(time.time())}'
    state={'run_id':run_id,'scenario_id':scenario_id,'messages':[{'role':'system','content':'You are an AGENT. Use the available tools to verify orders, consult policy, calculate refunds and act safely.'},{'role':'user','content':user_message}],
           'tokens_used':0,'llm_calls':0,'steps':[],'executed_tools':[]}
    _save_checkpoint(run_id,scenario_id,state['messages'],0,0,[],[])
    return _continue_agent(state,max_iterations,token_limit,time_limit_seconds,crash_after_tool_count)

def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
    state=json.loads((STATE/f'{run_id}.json').read_text(encoding='utf-8'))
    return _continue_agent(state,max_iterations,token_limit,time_limit_seconds,None)

print('PART D — POLICY RAG')
print('Overlap chunks:',len(CHUNKS))
rag_overlap=evaluate_policy(10); rag_zero=evaluate_policy(0)
print('RAG overlap evaluation:',rag_overlap)
print('RAG zero-overlap evaluation:',rag_zero)
# Ensure clean ledger for the reproducible scenario batch.
(OUT/'refund_ledger.jsonl').write_text('',encoding='utf-8')
scenarios=pd.read_csv(DATA/'raw'/'agent_scenarios.csv')
agent_results=[]; scenario_table=[]
for _,s in scenarios.iterrows():
    result=run_agent(s.message,s.scenario_id,run_id=f'scenario_{s.scenario_id}')
    refund=next((st['observation'].get('refund_inr') for st in result['steps'] if st['tool_name']=='calculate_refund'),0.0)
    item={'scenario_id':s.scenario_id,'status':result['status'],'final_answer':result['final_answer'],'llm_calls':result['llm_calls'],'tokens_used':result['tokens_used'],'steps':result['steps']}
    agent_results.append(item); scenario_table.append({'scenario_id':s.scenario_id,'status':result['status'],'refund_inr':refund,'llm_calls':result['llm_calls'],'tokens_used':result['tokens_used'],'expected_outcome':s.expected_outcome,'expected_refund_inr':float(s.expected_refund_inr)})
with (OUT/'agent_runs.jsonl').open('w',encoding='utf-8') as f:
    for item in agent_results: f.write(json.dumps(item)+'\n')

# Safe-stop demonstrations.
s1=scenarios.iloc[0]
token_demo=run_agent(s1.message,'S1_TOKEN_DEMO',token_limit=200,run_id='S1_TOKEN_DEMO')
time_demo=run_agent(s1.message,'S1_TIMEOUT_DEMO',time_limit_seconds=0,run_id='S1_TIMEOUT_DEMO')

# Crash after the second tool observation has been checkpointed; resume without replaying tools.
crash_run_id='S1_CRASH_RESUME'
try:
    run_agent(s1.message,'S1',run_id=crash_run_id,crash_after_tool_count=2)
except SimulatedCrash:
    pass
resumed=resume_agent(crash_run_id)

summary={'rag_overlap':rag_overlap,'rag_zero_overlap':rag_zero,'scenarios':scenario_table,
         'token_demo_status':token_demo['status'],'timeout_demo_status':time_demo['status'],
         'crash_resume_status':resumed['status'],'crash_resume_refund':next((x['observation'].get('refund_inr') for x in resumed['steps'] if x['tool_name']=='calculate_refund'),None),
         'crash_resume_tools':resumed['executed_tools'],'crash_resume_llm_calls':resumed['llm_calls'],
         'chunk_counts':{name:sum(c['source']==name for c in CHUNKS) for name in POLICY_FILES}}
(OUT/'part_d_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('\nAGENT SCENARIOS')
print(pd.DataFrame(scenario_table).to_string(index=False))
print('Token demo:',token_demo['status'],'Timeout demo:',time_demo['status'])
print('Crash/resume:',resumed['status'],resumed['executed_tools'])
print('\nPART D COMPLETE')