from transformers import AutoTokenizer, AutoModel
from model.modeling_chatglm import ChatGLMForConditionalGeneration
from model.configuration_chatglm import ChatGLMConfig
import time
import torch

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()
print(model.__class__)

prompt = "判断题。人类交通的发展可以追溯到古代，从最早的步行和动物运输开始。然而，随着技术的进步和全球化的推动，交通系统得以显著改进和扩展。铁路交通：19世纪的工业革命催生了铁路的发展。铁路系统的建立连接了城市和国家，促进了商业和人员流动。\
        蒸汽机车的发明为铁路交通带来了重大突破，后来的电力化和高速铁路进一步提升了交通效率和速度。汽车交通：20世纪初，汽车的发明引领了个人交通的革命。大规模的汽车生产和道路基础设施建设使得汽车成为人们日常出行的主要工具。\
        随着时间的推移，汽车的设计和性能不断改进，包括引入燃油效率更高的发动机和安全性能提升的创新。航空交通：20世纪初的飞机发明开创了航空交通的新纪元。航空技术的进步使得飞机能够更快、更远地飞行，带来了全球航空网络的形成。\
        民航业的发展使得长途旅行变得更加便捷，同时也推动了国际贸易和旅游业的增长。船舶交通：海洋运输一直是国际贸易的重要组成部分。从古代的帆船到现代的大型货船和油轮，船舶技术的不断发展使得海上运输更加高效和可靠。\
        公共交通：随着城市化的加速和交通拥堵问题的出现，公共交通系统得到了广泛的关注和发展。\
        地铁、轻轨、公交车和电车等公共交通工具的建设和改进，为城市居民提供了便捷、环保的出行选择。同时，共享交通和出租车服务的兴起也为人们提供了灵活的选择。\
        汽车的发展是人类交通史上的重大里程碑之一。从最早的蒸汽驱动汽车到现代电动汽车的兴起，汽车技术经历了巨大的变革和进步。以下是关于汽车发展的一些主要方面：\
        起源和早期发展：汽车的起源可以追溯到18世纪末和19世纪初。早期的汽车多采用蒸汽引擎，其中最著名的是尼古拉斯·约瑟夫·庞巴迪的蒸汽马车。\
        然而，随着内燃机的发明，汽油和柴油驱动的汽车逐渐取代了蒸汽驱动的汽车。大规模生产和亨利·福特：20世纪初，亨利·福特引领了汽车产业的革命。他的创新包括流水线生产和大规模制造模式，使得汽车生产变得高效且成本降低。\
        福特于1908年推出了著名的“T型车”，使汽车普及化，为大众交通带来了革命性的变化。设计和性能改进：汽车制造商在设计和性能方面进行了持续的改进。引入了气动外形设计、更高效的发动机和先进的底盘技术，提高了汽车的燃油经济性、安全性和驾驶体验。\
        创新的材料和制造工艺使得汽车更轻便、坚固和环保。飞机的发展自人类向天空展翅飞翔的梦想诞生以来，已经经历了令人瞩目的进步和变革。从最早的飞行器原型到今天的超音速喷气式客机和先进的无人机技术，飞机已经成为人类生活和全球经济的重要组成部分。\
        飞机的历史可以追溯到公元前5世纪，古希腊哲学家阿基米德设计了一种被称为“阿基米德螺旋”或“空气螺旋”的飞行器原型。然而，真正的飞机发展始于18世纪末和19世纪初，当时许多先驱者开始研究和试验各种飞行原理。\
        其中最为著名的是莱特兄弟，他们于1903年成功飞行了世界上第一架受人操纵的飞机，标志着现代航空的开始。自那时以来，飞机的发展取得了巨大的进步。早期的飞机主要采用螺旋桨作为动力源，而在第一次世界大战期间，飞机被广泛运用于军事行动。到了20世纪30年代，喷气式发动机的发明引领了飞机技术的新时代。\
        1949年，英国的“喷气式飞跃”实现了世界上第一次喷气式飞机的商业航班，标志着喷气时代的来临。在接下来的几十年里，飞机的设计和技术不断突破。喷气式客机的速度和载客能力大大提高，航空业迅速发展。随着航空工程的进步，涡轮螺旋桨飞机、超音速飞机和宽体客机相继问世。\
        20世纪60年代，喷气式客机的规模进一步扩大，波音747问世，成为当时最大的客机。此后，空中客车公司也推出了自己的宽体客机系列，如空中客车A380。\
        自人类掌握航海技术以来，船舶一直是人类探索和贸易的重要工具。从最早的木制划船到如今的现代化巨轮，轮船的发展经历了漫长而精彩的历程。\
        古代文明如古埃及、古希腊和古罗马都有自己的船舶建造技术和航行知识。然而，真正的轮船发展开始于19世纪。\
        蒸汽船的出现被视为轮船发展的重要里程碑。早期的蒸汽船使用蒸汽机驱动船轮，取代了传统的风帆和桨。第一艘商业化的蒸汽船是由美国工程师罗伯特·弗尔顿于1807年设计和建造的克莱蒙特号，它在哈德逊河上进行了首次试航。\
        这一创举引发了全球范围内对蒸汽船的兴趣，并催生了大量的蒸汽船公司和航线。随着工业革命的到来，蒸汽船的发展进一步加速。蒸汽机的改进和铁质船体的使用使得轮船的速度、负载能力和舒适度大大提高。\
        在19世纪中叶和下半叶，大西洋横渡成为蒸汽船的主要航线之一，远洋航行变得更加便捷和可靠。然而，随着内燃机技术的进步和石油的广泛应用，蒸汽船逐渐被内燃机驱动的船舶取代。内燃机船舶采用柴油发动机作为动力源，具有更高的效率和更低的运营成本。\
        20世纪初，柴油船成为主导船舶市场的力量并得到发展。基于历史知识，考虑并判断以上对于近现代交通工具发展的描述是否正确，仅仅回答“正确”或“错误”二字。"

configuration = model.config
for i in range(configuration.num_layers):
    model.transformer.layers[i].mlp.dense_h_to_4h_act.weight = model.transformer.layers[i].mlp.dense_h_to_4h.weight
    del model.transformer.layers[i].mlp.dense_h_to_4h.weight
    model.transformer.layers[i].mlp.dense_h_to_4h_act.bias = model.transformer.layers[i].mlp.dense_h_to_4h.bias
    del model.transformer.layers[i].mlp.dense_h_to_4h.bias
    torch.cuda.empty_cache()
response, history = model.chat(tokenizer, prompt, history=[])
print(response)
print("======================================")
start = time.perf_counter()
torch.cuda.cudart().cudaProfilerStart()
response, history = model.chat(tokenizer, prompt, history=[])
torch.cuda.cudart().cudaProfilerStop()
print(response)
print("======================================")
print("Time: ", time.perf_counter() - start)
# configuration = ChatGLMConfig(
#     bos_token_id=130004, 
#     eos_token_id=130005, 
#     mask_token_id=130000, 
#     gmask_token_id=130001,
#     pad_token_id=3,
#     use_cache=True,
#     vocab_size=130528,
#     model_type="chatglm",
#     torch_dtype="float16"
# )

# dialogues = 10
# history = []
# for j in range(dialogues):
#     response, history = model.chat(tokenizer, "北京是一个", history=[])
#     print(response)
#     total = 0
#     for (round_id, i) in enumerate(model.times):
#         if round_id > 0:
#             total += i
        #print(f"Round {round_id}: {i:.6f}s")
    # print(f"Dialogue: {j}")
    # print(f"Prefill: {model.times[0]:.6f}s")
    # print(f"Decode: {total:.6f}s | {round_id} rounds | {(total/round_id):.6f} s/round")
    # model.times = []
#for (param_id, p) in enumerate(model.transformer.layers[0].parameters()):
#    if p.requires_grad:
#        print(param_id, p.shape)
        #model.transformer.layers[1].parameters()[param_id] = p.half().cuda()

"""
transformer_layers = model.transformer.layers
#print(transformer_layers[1].__class__)

configuration = ChatGLMConfig(
    bos_token_id=130004, 
    eos_token_id=130005, 
    mask_token_id=130000, 
    gmask_token_id=130001,
    pad_token_id=3,
    use_cache=True,
    vocab_size=130528,
    model_type="chatglm",
    torch_dtype="float16"
)
#configuration = model.config
model = model.cpu()
new_model = ChatGLMForConditionalGeneration(configuration)
#print(new_model.transformer.layers[1].mlp.dense_4h_to_h.bias)
#print(model.transformer.layers[0].mlp.dense_4h_to_h.bias)
new_model.transformer.word_embeddings.weight = model.transformer.word_embeddings.weight
new_model.lm_head.weight = model.lm_head.weight
for i in range(configuration.num_layers):
    new_model.transformer.layers[i].input_layernorm.weight = model.transformer.layers[i].input_layernorm.weight
    new_model.transformer.layers[i].input_layernorm.bias = model.transformer.layers[i].input_layernorm.bias
    new_model.transformer.layers[i].attention.query_key_value.weight = model.transformer.layers[i].attention.query_key_value.weight
    new_model.transformer.layers[i].attention.query_key_value.bias = model.transformer.layers[i].attention.query_key_value.bias
    new_model.transformer.layers[i].attention.dense.weight = model.transformer.layers[i].attention.dense.weight
    new_model.transformer.layers[i].attention.dense.bias = model.transformer.layers[i].attention.dense.bias
    new_model.transformer.layers[i].post_attention_layernorm.weight = model.transformer.layers[i].post_attention_layernorm.weight
    new_model.transformer.layers[i].post_attention_layernorm.bias = model.transformer.layers[i].post_attention_layernorm.bias
    new_model.transformer.layers[i].mlp.dense_h_to_4h.weight = model.transformer.layers[i].mlp.dense_h_to_4h.weight
    new_model.transformer.layers[i].mlp.dense_h_to_4h.bias = model.transformer.layers[i].mlp.dense_h_to_4h.bias
    new_model.transformer.layers[i].mlp.dense_4h_to_h.weight = model.transformer.layers[i].mlp.dense_4h_to_h.weight
    new_model.transformer.layers[i].mlp.dense_4h_to_h.bias = model.transformer.layers[i].mlp.dense_4h_to_h.bias
new_model.half().cuda()
new_model = new_model.eval()
response, history = new_model.chat(tokenizer, "北京是一个", history=[])
print(response)
"""