from math import sqrt
import torch  
import torch.nn as nn  
from einops import rearrange
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer
from layers.Embed import PatchEmbedding  
import transformers  
from layers.StandardNorm import Normalize  
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, moving_avg
  
transformers.logging.set_verbosity_error()  
from accelerate import Accelerator, DeepSpeedPlugin  
from accelerate import DistributedDataParallelKwargs  
transformers.logging.set_verbosity_error()  
  
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)  
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])  
  
class FlattenHead(nn.Module):  
    def __init__(self, n_vars, nf, target_window, head_dropout=0):  
        super().__init__()  
        self.n_vars = n_vars  
        self.flatten = nn.Flatten(start_dim=-2)  
        self.linear = nn.Linear(nf, target_window)  
        self.dropout = nn.Dropout(head_dropout)  
  
    def forward(self, x):  
        x = self.flatten(x)  
        x = self.linear(x)  
        x = self.dropout(x)  
        return x  
  
  
class Model(nn.Module):  
  
    def __init__(self, configs, patch_len=16, stride=8):  
        super(Model, self).__init__()  
        self.task_name = configs.task_name  
        self.pred_len = configs.pred_len  
        self.seq_len = configs.seq_len  
        self.d_ff = configs.d_ff  
        self.top_k = 5  
        self.d_llm = configs.llm_dim  
        self.patch_len = configs.patch_len  
        self.stride = configs.stride  
        self.output_attn_map = configs.output_attn_map  
        self.decomp_method = configs.decomp_method  
        self.decomp_level = configs.decomp_level  
        # Decomp  
        kernel_size = configs.moving_avg  
        self.decomp = series_decomp(kernel_size)  
  
        if configs.llm_model == 'LLAMA':  
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')  
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')  
            self.llama_config.num_hidden_layers = configs.llm_layers  
            self.llama_config.output_attentions = True  
            self.llama_config.output_hidden_states = True  
            try:  
                self.llm_model = LlamaModel.from_pretrained(  
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",  
                    'huggyllama/llama-7b',  
                    trust_remote_code=True,  
                    local_files_only=True,  
                    config=self.llama_config,  
                    # load_in_4bit=True  
                )  
            except EnvironmentError:  # downloads model from HF is not already done  
                print("Local model files not found. Attempting to download...")  
                self.llm_model = LlamaModel.from_pretrained(                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",  
                    'huggyllama/llama-7b',  
                    trust_remote_code=True,  
                    local_files_only=False,  
                    config=self.llama_config,  
                    # load_in_4bit=True  
                )  
            try:  
                self.tokenizer = LlamaTokenizer.from_pretrained(  
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",  
                    'huggyllama/llama-7b',  
                    trust_remote_code=True,  
                    local_files_only=True  
                )  
            except EnvironmentError:  # downloads the tokenizer from HF if not already done  
                print("Local tokenizer files not found. Atempting to download them..")  
                self.tokenizer = LlamaTokenizer.from_pretrained(  
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",  
                    'huggyllama/llama-7b',  
                    trust_remote_code=True,  
                    local_files_only=False  
                )  
        elif configs.llm_model == 'GPT2':  
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')  
  
            self.gpt2_config.num_hidden_layers = configs.llm_layers  
            self.gpt2_config.output_attentions = True  
            self.gpt2_config.output_hidden_states = True  
            try:  
                self.llm_model = GPT2Model.from_pretrained(  
                    'openai-community/gpt2',  
                    trust_remote_code=True,  
                    local_files_only=True,  
                    config=self.gpt2_config,  
                )  
            except EnvironmentError:  # downloads model from HF is not already done  
                print("Local model files not found. Attempting to download...")  
                self.llm_model = GPT2Model.from_pretrained(  
                    'openai-community/gpt2',  
                    trust_remote_code=True,  
                    local_files_only=False,  
                    config=self.gpt2_config,  
                )  
  
            try:  
                self.tokenizer = GPT2Tokenizer.from_pretrained(  
                    'openai-community/gpt2',  
                    trust_remote_code=True,  
                    local_files_only=True  
                )  
            except EnvironmentError:  # downloads the tokenizer from HF if not already done  
                print("Local tokenizer files not found. Atempting to download them..")  
                self.tokenizer = GPT2Tokenizer.from_pretrained(  
                    'openai-community/gpt2',  
                    trust_remote_code=True,  
                    local_files_only=False  
                )  
        elif configs.llm_model == 'BERT':  
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')  
  
            self.bert_config.num_hidden_layers = configs.llm_layers  
            self.bert_config.output_attentions = True  
            self.bert_config.output_hidden_states = True  
            try:  
                self.llm_model = BertModel.from_pretrained(  
                    'google-bert/bert-base-uncased',  
                    trust_remote_code=True,  
                    local_files_only=True,  
                    config=self.bert_config,  
                )  
            except EnvironmentError:  # downloads model from HF is not already done  
                print("Local model files not found. Attempting to download...")  
                self.llm_model = BertModel.from_pretrained(  
                    'google-bert/bert-base-uncased',  
                    trust_remote_code=True,  
                    local_files_only=False,  
                    config=self.bert_config,  
                )  
  
            try:  
                self.tokenizer = BertTokenizer.from_pretrained(  
                    'google-bert/bert-base-uncased',  
                    trust_remote_code=True,  
                    local_files_only=True  
                )  
            except EnvironmentError:  # downloads the tokenizer from HF if not already done  
                print("Local tokenizer files not found. Atempting to download them..")  
                self.tokenizer = BertTokenizer.from_pretrained(  
                    'google-bert/bert-base-uncased',  
                    trust_remote_code=True,  
                    local_files_only=False  
                )  
        else:  
            raise Exception('LLM model is not defined')  
  
        if self.tokenizer.eos_token:  
            self.tokenizer.pad_token = self.tokenizer.eos_token  
        else:  
            pad_token = '[PAD]'  
            self.tokenizer.add_special_tokens({'pad_token': pad_token})  
            self.tokenizer.pad_token = pad_token  
  
        for param in self.llm_model.parameters():  
            param.requires_grad = False  
  
        if configs.prompt_domain:  
            self.description = configs.content  
        else:  
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'  
  
        self.dropout = nn.Dropout(configs.dropout)  
  
        self.patch_embedding = PatchEmbedding(  
            configs.d_model, self.patch_len, self.stride, configs.dropout)  
  
        self.word_embeddings = self.llm_model.get_input_embeddings().weight  
        self.vocab_size = self.word_embeddings.shape[0]  
        self.num_tokens = 1000  
        self.num_tokens_trend = 10  
        self.num_tokens_seasonal = 100  
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)  
        self.mapping_layer_trend = nn.Linear(self.vocab_size, self.num_tokens_trend)  
        self.mapping_layer_seasonal = nn.Linear(self.vocab_size, self.num_tokens_seasonal)  
        self.in_layer_trend = nn.Linear(configs.patch_len, configs.d_model)  
        self.in_layer_seasonal = nn.Linear(configs.patch_len, configs.d_model)  
        self.in_layer_residual = nn.Linear(configs.patch_len, configs.d_model)  
        self.map_trend = nn.Linear(configs.seq_len, configs.seq_len)  
        self.map_season  = nn.Sequential(  
            nn.Linear(configs.seq_len, 4*configs.seq_len),  
            nn.ReLU(),  
            nn.Linear(4*configs.seq_len, configs.seq_len)  
        )  
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))  
        self.moving_avg = moving_avg(kernel_size, stride=self.stride)  
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)  
  
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)  
        self.head_nf = self.d_ff * self.patch_nums  
  
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':  
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,  
                                                 head_dropout=configs.dropout)  
        else:  
            raise NotImplementedError  
  
        self.normalize_layers = Normalize(configs.enc_in, affine=False)  
  
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec , seq_trend, seq_seasonal, seq_resid, mask=None):  
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':  
            dec_out, attn_map_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, seq_trend, seq_seasonal, seq_resid)  
            #accelerator.print("args.output_attn_map is " + str(self.output_attn_map))  
            if self.output_attn_map:  
                return dec_out, attn_map_list#attn_map in order of seasonal, trend, residual  
            else:  
                return dec_out[:, -self.pred_len:, :]  
        return None  
  
    def get_norm(self, x, d = 'norm'):  
        # if d == 'norm':  
        means = x.mean(1, keepdim=True).detach()  
        x = x - means  
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach()  
        x /= stdev  
  
        return x, means, stdev  
  
    def get_patch(self, x):  
        #x = rearrange(x, 'b l m -> b m l')  
        n_vars = x.shape[1]  
        x = self.padding_patch_layer(x) # 4, 1, 420  
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) #4,1, 64, 16  
        x = rearrange(x, 'b m n p -> (b m) n p') # 4, 64, 16  
  
        return x, n_vars  
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, seq_trend, seq_seasonal, seq_resid):  
  
        x_enc = self.normalize_layers(x_enc, 'norm')  
  
        B, T, N = x_enc.size()  
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)  
  
        min_values = torch.min(x_enc, dim=1)[0]  
        max_values = torch.max(x_enc, dim=1)[0]  
        medians = torch.median(x_enc, dim=1).values  
        lags = self.calcute_lags(x_enc)  
        trends = x_enc.diff(dim=1).sum(dim=1)  
  
        prompt = []  
        for b in range(x_enc.shape[0]):  
            min_values_str = str(min_values[b].tolist()[0])  
            max_values_str = str(max_values[b].tolist()[0])  
            median_values_str = str(medians[b].tolist()[0])  
            lags_values_str = str(lags[b].tolist())  
            prompt_ = (  
                f"<|start_prompt|>Dataset description: {self.description}"  
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "  
                "Input statistics: "  
                f"min value {min_values_str}, "  
                f"max value {max_values_str}, "  
                f"median value {median_values_str}, "  
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "  
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"  
            )  
  
            prompt.append(prompt_)  
  
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()  
  
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids  
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)  
  
        source_embeddings_original = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        source_embeddings_trend = self.mapping_layer_trend(self.word_embeddings.permute(1, 0)).permute(1, 0)  
        source_embeddings_seasonal = self.mapping_layer_seasonal(self.word_embeddings.permute(1, 0)).permute(1, 0)  
        x_enc = x_enc.permute(0, 2, 1).contiguous()  
        if self.decomp_level == 1:#TimeLLM  
            enc_out, n_vars = self.patch_embedding(x_enc)  
            components = {'original': enc_out}  
        elif self.decomp_level == 2:  
            seasonal_init, trend_init = self.decomp(x_enc)  
            seasonal_out, n_vars = self.patch_embedding(seasonal_init)  
            trend_out, n_vars = self.patch_embedding(trend_init)  
            components = {'seasonal_resid': seasonal_out, 'trend': trend_out}
        elif self.decomp_method == 'STL':  
            trend_init, means_trend, stdev_trend = self.get_norm(seq_trend.permute(0, 2, 1).contiguous())
            seasonal_init, means_seasonal, stdev_seasonal = self.get_norm(seq_seasonal.permute(0, 2, 1).contiguous())
            residual_init, means_seasonal, stdev_seasonal = self.get_norm(seq_resid.permute(0, 2, 1).contiguous())
            seasonal_out, n_vars = self.patch_embedding(seasonal_init)  
            trend_out, n_vars = self.patch_embedding(trend_init)  
            residual_out, n_vars = self.patch_embedding(residual_init)  
            components = {'seasonal': seasonal_out, 'trend': trend_out, 'residual': residual_out}  
        elif self.decomp_method == 'TEMPO':  
            trend_init = self.moving_avg(x_enc)  
            trend_init = self.map_trend(trend_init)  
            seasonal_init = x_enc - trend_init  
            # print(season_local.squeeze().shape)  
            seasonal_init = self.map_season(seasonal_init)  
            residual_init = x_enc - trend_init - seasonal_init  
            trend_out, n_vars = self.get_patch(trend_init)  
            seasonal_out, n_vars = self.get_patch(seasonal_init)  
            residual_out, n_vars = self.get_patch(residual_init)  
            trend_out = self.in_layer_trend(trend_out)  
            seasonal_out = self.in_layer_seasonal(seasonal_out)  
            residual_out = self.in_layer_residual(residual_out)  
            components = {'seasonal': seasonal_out, 'trend': trend_out, 'residual': residual_out}  
        ##do patch reprogramming for both of seasonal and trend  
        dec_components = []  
        attn_map_list = []  
        for k,v in components.items():  
            if k == 'seasonal':  
                source_embeddings = source_embeddings_seasonal  
            elif k == 'trend':  
                source_embeddings = source_embeddings_trend  
            elif k == 'residual' or 'original' or 'seasonal_resid':
                source_embeddings = source_embeddings_original
            components_out, attn_map = self.reprogramming_layer(v, source_embeddings, source_embeddings)  
            llama_components_out = torch.cat([prompt_embeddings, components_out], dim=1)  
            dec_components_out = self.llm_model(inputs_embeds=llama_components_out).last_hidden_state  
            dec_components_out = dec_components_out[:, :, :self.d_ff]  
            dec_components_out = torch.reshape(  
                dec_components_out, (-1, n_vars, dec_components_out.shape[-2], dec_components_out.shape[-1]))  
            dec_components_out = dec_components_out.permute(0, 1, 3, 2).contiguous()  
            dec_components_out = self.output_projection(dec_components_out[:, :, :, -self.patch_nums:])  
            dec_components_out = dec_components_out.permute(0, 2, 1).contiguous()  
            dec_components.append(dec_components_out)  
            attn_map_list.append(attn_map)  
        dec_components_out = sum(dec_components)  
        dec_components_out = self.normalize_layers(dec_components_out, 'denorm')  
        dec_out = dec_components_out  
  
        return dec_out, attn_map_list  
  
    def calcute_lags(self, x_enc):  
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)  
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)  
        res = q_fft * torch.conj(k_fft)  
        corr = torch.fft.irfft(res, dim=-1)  
        mean_value = torch.mean(corr, dim=1)  
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)  
        return lags  
  
  
class ReprogrammingLayer(nn.Module):  
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):  
        super(ReprogrammingLayer, self).__init__()  
  
        d_keys = d_keys or (d_model // n_heads)  
  
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)  
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)  
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)  
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)  
        self.n_heads = n_heads  
        self.dropout = nn.Dropout(attention_dropout)  
  
    def forward(self, target_embedding, source_embedding, value_embedding):  
        B, L, _ = target_embedding.shape  
        S, _ = source_embedding.shape  
        H = self.n_heads  
  
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)  
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)  
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)  
  
        out, attn = self.reprogramming(target_embedding, source_embedding, value_embedding)  
  
        out = out.reshape(B, L, -1)  
  
        return self.out_projection(out), attn  
  
    def reprogramming(self, target_embedding, source_embedding, value_embedding):  
        B, L, H, E = target_embedding.shape  
  
        scale = 1. / sqrt(E)  
  
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)  
  
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)  
  
        return reprogramming_embedding, A