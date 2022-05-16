from testapp import app
from flask import render_template, request, make_response, Markup
from testapp.auto_story import CharDataset, Args, GPT, LTConfig, LanguageTrainer, My_story
import os
def prepare_response(data):#セキュリティ対策
    response = make_response(data)
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains' #HTTP Strict Transport Security（HSTS）HTTPではなくHTTPSで接続するようブラウザに要求します。
    #response.headers['Content-Security-Policy'] = 'default-src \'self\'' #ページ上で読み込むリソースの読み込み先をホワイトリスト形式で指定します。
    response.headers['X-Content-Type-Options'] = 'nosniff' #レスポンスヘッダで指定したcontent typeに絶対に従いなさいという指示
    response.headers['X-Frame-Options'] = 'SAMEORIGIN' #HTMLのframeやiframeで呼び出されることを許可する範囲を指定, 一切禁止：DENY, 特定のドメインを許可：ALLOW-FROM https://example.com/
    response.headers['X-XSS-Protection'] = '1; mode=block' #ブラウザのセキュリティ機能を利用してXSS攻撃を抑えるものです。しかしながら完全ではなく、また思わぬ誤作動の危険性もあります。とりあえず設定しておけばいいというものではない。
    return response

STEPS=100 #ここで取り出す文の長さを調節する。だいたいこの数値分の文字数出力する。
block_size = 128 #128から変更できない # コンテクストの長さ
dire = os.getcwd()
work_dir = dire+'/testapp/world_model_lec6/'
# 事前学習用データセット. ファイルは1.1MB程度です.
text = open(work_dir+'shakespeare.txt', 'r').read()
train_dataset = CharDataset(text, block_size)
args = Args({
        'vocab_size': train_dataset.vocab_size,
        'block_size': train_dataset.block_size,
        'n_layer': 8,
        'n_head': 8,
        'n_embd': 512,
        'rl': False, # Trajectory Transformerで行う処理を飛ばす
    })
model=GPT(args) # あとでtrainerがモデルをGPUに移してくれます.
    # LanguageTrainerをインスタンス化し, 訓練を開始します.
    # batch_sizeを512にするとメモリ不足になります.
tconf = LTConfig(max_epochs=2, batch_size=256, learning_rate=6e-4,
                lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*block_size,
                num_workers=2)
trainer = LanguageTrainer(model, train_dataset, None, tconf)

#独自のフィルター設定
@app.template_filter( "linebreaksbr" )
def func_linebreaksbr( linebreaksbr ) :
    return Markup(linebreaksbr.replace("\n", "<br />\n"))

@app.template_filter( "deln" )
def func_linebreaksbr( deln ) :
    return Markup(deln.replace("\n", ""))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method=='GET':
        response_body = render_template('testapp/index.html')
        response = prepare_response(response_body)
        return response
    if request.method=='POST':
        print("Posted!!!")
        req1 = request.form['input']
        print("sub:",req1)
        DEVICE = 'cpu'
    try:
        story = My_story(model=model, train_dataset=train_dataset, trainer=trainer,Input=req1, Work_dir=work_dir, GPU=False,STEPS=STEPS)
    except IndexError:
        print("IndexError")
        return 'Error occurred "Blank". Please fill out.'
    except KeyError:
        print("KeyError")
        return 'Error occured "Not half-width alphanumeric characters".Please fill out using half-width alphanumeric characters.'
    except Exception as e:
        print("Error:",e)
        #import traceback
        #traceback.print_exc()
        response_body = render_template('testapp/error.html',DEVICE=DEVICE)
        response = prepare_response(response_body)
        return response
    response_body = render_template('testapp/result.html',req=req1,story=story)
    response = prepare_response(response_body)
    return response


@app.route('/GPU', methods=['GET', 'POST'])
def index_GPU():
    if request.method=='GET':
        response_body = render_template('testapp/index-GPU.html')
        response = prepare_response(response_body)
        return response
    if request.method=='POST':
        print("Posted!!!")
        req1 = request.form['input']
        DEVICE='gpu'
        try:
            story = My_story(model=model, train_dataset=train_dataset, trainer=trainer,Input=req1, Work_dir=work_dir, GPU=True)
        except IndexError:
            return 'Error occurred "Blank". Please fill out.'
        except KeyError:
            return 'Error occured "Not half-width alphanumeric characters".Please fill out using half-width alphanumeric characters.'
        except:
            response_body = render_template('testapp/error.html',DEVICE=DEVICE)
            response = prepare_response(response_body)
            return response
        response_body = render_template('testapp/result-GPU.html',req=req1,story=story)
        response = prepare_response(response_body)
        return response

