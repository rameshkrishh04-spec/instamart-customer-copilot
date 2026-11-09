"""Part C: privacy-safe, guarded, retryable and budget-aware ticket triage."""
from pathlib import Path
import json, re, random, time
import pandas as pd
import mock_llm
from sklearn.metrics import confusion_matrix, accuracy_score

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
TICKETS=pd.read_csv(ROOT/'data/raw/tickets.csv',keep_default_na=False)
SEG=pd.read_csv(OUT/'customer_segments.csv')
SEG['is_premium']=SEG['is_premium'].astype(str).str.lower().isin(['true','1'])
ALLOWED={'billing','delivery','technical','fraud'}
MAX_INPUT_TOKENS=400; BASE_DELAY=0.05; MAX_RETRIES=3
EMAIL_RE=r'[\w.+-]+@[\w-]+\.[\w.]+'
PHONE_RE=r'\b[6-9]\d{9}\b'
ORDER_RE=r'ORD-\d{6}'

ZERO_SHOT_PROMPT=("You are a CLASSIFY support-ticket classifier. Classify each ticket into exactly one allowed category: billing, delivery, technical, fraud. "
                  "Return JSON only with the form {\"category\": \"...\", \"reason\": \"...\"}. Do not add markdown or other text.")
FEW_SHOT_PROMPT=("You are a CLASSIFY support-ticket classifier. Classify each ticket into exactly one allowed category: billing, delivery, technical, fraud. "
                 "Return JSON only with the form {\"category\": \"...\", \"reason\": \"...\"}. Do not add markdown or other text.\n"
                 "EXAMPLES:\n"
                 "Ticket: Someone used my account without permission; this is fraud.\nOutput: {\"category\":\"fraud\",\"reason\":\"unauthorized account activity\"}\n"
                 "Ticket: The app crashes when I open the cart.\nOutput: {\"category\":\"technical\",\"reason\":\"app crash\"}\n"
                 "Ticket: My delivery arrived late.\nOutput: {\"category\":\"delivery\",\"reason\":\"late delivery\"}")

def redact(text):
    counts={}
    text,n=re.subn(EMAIL_RE,'[EMAIL]',text); counts['emails']=n
    text,n=re.subn(PHONE_RE,'[PHONE]',text); counts['phones']=n
    text,n=re.subn(ORDER_RE,'[ORDER_ID]',text); counts['order_ids']=n
    return text,counts

def validate_reply(text):
    try:
        obj=json.loads(text)
        if isinstance(obj,dict) and obj.get('category') in ALLOWED:
            return obj['category'],'classified_json'
        return '', 'flagged_invalid_output'
    except (json.JSONDecodeError,TypeError):
        words=re.findall(r'[a-z]+',(text or '').lower())
        found=[w for w in words if w in ALLOWED]
        unique=set(found)
        if len(unique)==1: return next(iter(unique)),'classified_keyword'
        return '', 'flagged_invalid_output'

def retry_chat(messages,ticket_id,rng):
    retries=0
    while True:
        try:
            return mock_llm.chat(messages),retries
        except mock_llm.TransientLLMError:
            if retries>=MAX_RETRIES: raise
            wait=BASE_DELAY*(2**retries)*rng.uniform(0.9,1.15)
            print(f"RETRY ticket={ticket_id} retry={retries} wait={wait:.5f}s")
            time.sleep(wait); retries+=1

def route_result(status,category,customer_id):
    if status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}:
        return 'NORMAL','human_review'
    if category=='fraud': return 'HIGH','security_team'
    row=SEG.loc[SEG.customer_id.eq(customer_id)]
    if not row.empty and bool(row.iloc[0].is_premium): return 'HIGH','premium_desk'
    return 'NORMAL','support_queue'

def run_pipeline(prompt,token_budget=None,run_name='run'):
    mock_llm.reset(); rng=random.Random(42)
    running_tokens=0; retry_count=0; rows=[]; truth=[]; pred=[]
    redaction_totals={'emails':0,'phones':0,'order_ids':0}
    for _,ticket in TICKETS.iterrows():
        text=ticket['text'] or ''
        clean,counts=redact(text)
        for k,v in counts.items(): redaction_totals[k]+=v
        # Assert that the three sensitive patterns no longer match the redacted text.
        assert re.search(EMAIL_RE,clean) is None
        assert re.search(PHONE_RE,clean) is None
        assert re.search(ORDER_RE,clean) is None
        status=''; category=''; used=0
        if not text.strip():
            status='skipped_empty'
        elif mock_llm.est_tokens(text)>MAX_INPUT_TOKENS:
            status='flagged_long'
        elif token_budget is not None and running_tokens>=token_budget:
            status='budget_exhausted'
        else:
            messages=[{'role':'system','content':prompt},{'role':'user','content':clean}]
            # The user message is exactly the redacted ticket text.
            assert messages[-1]['content']==clean
            response,retries=retry_chat(messages,ticket['ticket_id'],rng)
            retry_count+=retries
            used=int(response.usage.total_tokens); running_tokens+=used
            category,status=validate_reply(response.choices[0].message.content)
            if status in {'classified_json','classified_keyword'}:
                truth.append(ticket['true_category']); pred.append(category)
        priority,route=route_result(status,category,ticket['customer_id'])
        rows.append({'ticket_id':ticket['ticket_id'],'customer_id':ticket['customer_id'],'status':status,
                     'category':category,'segment_name':(SEG.loc[SEG.customer_id.eq(ticket['customer_id']),'segment_name'].iloc[0]
                     if not SEG.loc[SEG.customer_id.eq(ticket['customer_id'])].empty else ''),
                     'priority':priority,'route':route,'tokens_used':used})
    df=pd.DataFrame(rows)
    classified=len(truth)
    summary={'run':run_name,'rows':len(df),'status_counts':df.status.value_counts().to_dict(),
             'classified':classified,'accuracy':float(accuracy_score(truth,pred)) if classified else None,
             'tokens_total':running_tokens,'tokens_per_classified':running_tokens/classified if classified else None,
             'retry_count':retry_count,'redaction_counts':redaction_totals,
             'confusion_matrix':confusion_matrix(truth,pred,labels=['billing','delivery','technical','fraud']).tolist() if classified else []}
    return df,summary

print('PART C — SUPPORT TICKET TRIAGE')
# Two independent full runs; reset occurs inside run_pipeline immediately before each.
zero_df,zero_summary=run_pipeline(ZERO_SHOT_PROMPT,None,'zero_shot')
few_df,few_summary=run_pipeline(FEW_SHOT_PROMPT,None,'few_shot')
# Budget demonstration: use 6000 as required; lower to 5000 only if prompt length avoids exhaustion.
budget_df,budget_summary=run_pipeline(ZERO_SHOT_PROMPT,6000,'zero_shot_budget_6000')
if budget_summary['status_counts'].get('budget_exhausted',0)==0:
    budget_df,budget_summary=run_pipeline(ZERO_SHOT_PROMPT,5000,'zero_shot_budget_5000')

# Final few-shot run with generous budget and exact routing rules.
final_df,final_summary=run_pipeline(FEW_SHOT_PROMPT,100000,'few_shot_final')
final_df.to_csv(OUT/'triage_results.csv',index=False)
budget_df.to_csv(OUT/'triage_budget_demo.csv',index=False)
summary={'zero_shot':zero_summary,'few_shot':few_summary,'budget_demo':budget_summary,'final':final_summary}
(OUT/'part_c_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('\nZERO-SHOT:',zero_summary)
print('\nFEW-SHOT:',few_summary)
print('\nBUDGET DEMO:',budget_summary)
print('\nFINAL TRIAGE STATUS COUNTS:',final_summary['status_counts'])
print('Redaction totals:',final_summary['redaction_counts'])
print('\nPART C COMPLETE')