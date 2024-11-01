from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Sampler
from torch.utils.tensorboard import SummaryWriter

from data_prepare.load_data import *
from data_prepare.prototypical_batch_sampler import *
from network import rescnn
from network.Cross_Att import *
from network.proto_att import *
from parse.parser_camp import get_parser
# from label_smooth import CE_Label_Smooth_Loss
from utils.gdd import gdd
from utils.prototypical_loss import *
from utils.early_stop import *
from network.model import *
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed = 42):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def init_optim(opt, model, epoch=None):
    '''
    Initialize optimizer
    '''
    optim = torch.optim.Adam(params=model.parameters(), lr=opt.learning_rate, weight_decay=0.01)
    return optim

def val(parser,data_dset, data_sampler, model,protonet):
    device = torch.device("cuda:" + str(parser.cuda) if torch.cuda.is_available else "cpu")
    model.eval()
    val_proto_loss = 0.0
    val_proto_acc = 0.0

    train_iter = iter(data_sampler["train_sampler"])
    train_eeg_dset = data_dset["train_eeg_dset"]
    train_eye_dset = data_dset["train_eye_dset"]

    val_iter = iter(data_sampler["val_sampler"])
    val_eeg_dset = data_dset["val_eeg_dset"]
    val_eye_dset = data_dset["val_eye_dset"]
    with torch.no_grad():
        for it in range(parser.iterations):
            source_indices = next(train_iter)
            source_eeg_batch, source_labels = train_eeg_dset[source_indices]
            source_eye_batch, _ = train_eye_dset[source_indices]

            val_indices = next(val_iter)
            val_eeg_batch, val_labels = val_eeg_dset[val_indices]
            val_eye_batch, _ = val_eye_dset[val_indices]
            # print(f"target_eeg_batch.shape:{target_eeg_batch.shape}")

            source_eeg_batch =  source_eeg_batch.to(device)
            source_eye_batch =  source_eye_batch.to(device)
            source_labels = source_labels.to(device)

            val_eeg_batch =  val_eeg_batch.to(device)
            val_eye_batch =  val_eye_batch.to(device)
            val_labels = val_labels.to(device)

            source_fusion = model(source_eeg_batch,source_eye_batch)
            val_fusion = model(val_eeg_batch, val_eye_batch)

            dist_tgt = protonet([source_fusion[0], val_fusion[0]], [source_labels, val_labels],
                                parser.classes_per_it_tgt, parser.num_support_tgt, parser.num_query_tgt,flag = 1)

            proto_loss, proto_acc, _ = prototypical_loss2(dist_tgt, parser.classes_per_it_tgt, parser.num_query_tgt, parser)

            val_proto_acc += proto_acc / parser.iterations
            val_proto_loss += proto_loss / parser.iterations


    return val_proto_loss, val_proto_acc

def test(parser,data_dset, data_sampler, model,protonet):
    device = torch.device("cuda:" + str(parser.cuda) if torch.cuda.is_available else "cpu")
    model.eval()

    test_proto_loss = 0.0
    test_proto_acc = 0.0

    train_iter = iter(data_sampler["train_sampler"])
    train_eeg_dset = data_dset["train_eeg_dset"]
    train_eye_dset = data_dset["train_eye_dset"]

    test_target_iter = iter(data_sampler["test_sampler"])
    test_eeg_dset = data_dset["test_eeg_dset"]
    test_eye_dset = data_dset["test_eye_dset"]

    with torch.no_grad():
        for it in range(parser.iterations):
            source_indices = next(train_iter)
            source_eeg_batch, source_labels = train_eeg_dset[source_indices]
            source_eye_batch, _ = train_eye_dset[source_indices]

            test_indices = next(test_target_iter)
            test_eeg_batch, test_labels = test_eeg_dset[test_indices]
            test_eye_batch, _ = test_eye_dset[test_indices]
            # print(f"target_eeg_batch.shape:{target_eeg_batch.shape}")

            source_eeg_batch =  source_eeg_batch.to(device)
            source_eye_batch =  source_eye_batch.to(device)
            source_labels = source_labels.to(device)

            test_eeg_batch =  test_eeg_batch.to(device)
            test_eye_batch =  test_eye_batch.to(device)
            test_labels = test_labels.to(device)

            source_fusion = model(source_eeg_batch, source_eye_batch)
            test_fusion = model(test_eeg_batch, test_eye_batch)

            dist_tgt = protonet([source_fusion[0], test_fusion[0]], [source_labels, test_labels],
                                parser.classes_per_it_tgt, parser.num_support_tgt, parser.num_query_tgt, flag=1)

            proto_loss, proto_acc, conf_matrix = prototypical_loss2(dist_tgt, parser.classes_per_it_tgt, parser.num_query_tgt, parser)

            test_proto_acc += proto_acc / parser.iterations
            test_proto_loss += proto_loss / parser.iterations


    return test_proto_loss, test_proto_acc, conf_matrix


# design hyper-parameters
eeg_input_dim = 256
eye_input_dim = 177
output_dim = 256
emotion_categories = 3

def main(parser,data_dset, data_sampler, writer,early_stopping):

    device = torch.device("cuda:" + str(parser.cuda) if torch.cuda.is_available else "cpu")
    # learning_rate = parser.learning
    set_seed(parser.seed)

    # Create the model
    model = MyModel(eeg_input_dim, eye_input_dim, output_dim)
    protonet = ProtoNet()

    # Use GPU

    model = model.to(device)
    protonet = protonet.to(device)

    params = list(model.parameters())
    # Optimizer
    optim = torch.optim.Adam(params=params, lr=parser.learning_rate, weight_decay=0.01)
    # # 使用学习率调度器
    # scheduler = StepLR(optim, step_size=10, gamma=0.5)

    interval = 1
    best_acc = 0.0
    tar_center = None
    print("----------Starting training the model----------")
    # Begin training
    for epoch in range(1, parser.epochs+1):
        print('========= Epoch: {} ========='.format(epoch))
        model.train()
        protonet.train()

        train_loss = 0
        train_gdd_loss = 0
        train_proto_loss_src = 0
        train_proto_loss_tgt = 0
        train_proto_acc_src = 0
        train_proto_acc_tgt = 0
        # correct = 0
        # count = 0

        train_source_iter = iter(data_sampler["source_sampler"])
        train_iter = iter(data_sampler["train_sampler"])
        source_eeg_dset = data_dset["source_eeg_dset"]
        source_eye_dset = data_dset["source_eye_dset"]
        train_eeg_dset = data_dset["train_eeg_dset"]
        train_eye_dset = data_dset["train_eye_dset"]

        for it in range(parser.iterations):
            # load source data and target data
            source_indices = next(train_source_iter)
            source_eeg_batch, source_labels = source_eeg_dset[source_indices]
            source_eye_batch, _ = source_eye_dset[source_indices]

            source_eeg_batch = source_eeg_batch.to(device)
            source_eye_batch = source_eye_batch.to(device)
            source_labels = source_labels.to(device)

            target_indices = next(train_iter)
            target_eeg_batch, target_labels = train_eeg_dset[target_indices]
            target_eye_batch, _ = train_eye_dset[target_indices]

            target_eeg_batch = target_eeg_batch.to(device)
            target_eye_batch = target_eye_batch.to(device)
            target_labels = target_labels.to(device)

            source_fusion = model(source_eeg_batch, source_eye_batch)
            target_fusion = model(target_eeg_batch, target_eye_batch)

            # gdd_loss_eeg = gdd(src_eeg, tgt_eeg)
            gdd_loss1 = gdd(source_fusion[0], target_fusion[0])
            # gdd_loss2 = gdd(source_fusion[1], target_fusion[1])
            # gdd_loss3 = gdd(source_fusion[2], target_fusion[2])
            # gdd_loss4 = gdd(source_fusion[3], target_fusion[3])
            # gdd_loss5 = gdd(source_fusion[4], target_fusion[4])

            dist_src = protonet(source_fusion[0], source_labels, parser.classes_per_it_src, parser.num_support_src, parser.num_query_src)
            proto_loss_src, proto_acc_src, _ = prototypical_loss2(dist_src, parser.classes_per_it_src, parser.num_query_src,parser)

            # dist_tgt = protonet(target_fusion[0], target_labels, parser.classes_per_it_tgt, parser.num_support_tgt, parser.num_query_tgt)
            # proto_loss_tgt, proto_acc_tgt = prototypical_loss2(dist_tgt, parser.classes_per_it_tgt, parser.num_query_tgt, parser)

            gamma = 2 / (1 + np.exp(-10 * (epoch) / (parser.epochs))) - 1
            # 定义参数
            a = 0.5  # 权重的缩放因子
            b = 1  # 对数函数的系数

            # 计算权重
            w = [a * np.log(b * i + 1) for i in range(1, 2)]

            # 打印结果
            w1 = w[0]

            # 归一化
            total_weight = w1
            w1 /= total_weight
            # w2 /= total_weight
            # w3 /= total_weight
            # w4 /= total_weight
            # w5 /= total_weight

            gdd_loss = gamma * (w1 * gdd_loss1)

            loss = proto_loss_src + gdd_loss

            # 更新参数
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss / parser.iterations
            train_gdd_loss += (gdd_loss) / parser.iterations
            train_proto_loss_src += proto_loss_src / parser.iterations
            train_proto_acc_src += proto_acc_src / parser.iterations

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('gdd_loss', train_gdd_loss, epoch)
        writer.add_scalar('proto_loss_src', train_proto_loss_src, epoch)
        writer.add_scalar('proto_acc_src', train_proto_acc_src, epoch)

        print('Train Loss: {:.6f}\tmmd_loss:{:.6f}\tproto_loss_src:{:.6f}\tproto_loss_tgt:{:.6f}\tproto_acc_src:{:.6f}\tproto_acc_tgt:{:.6f}'.
            format(train_loss, train_gdd_loss, train_proto_loss_src, train_proto_loss_tgt, train_proto_acc_src,
                   train_proto_acc_tgt))


        val_proto_loss, val_proto_acc = val(parser,data_dset,data_sampler,model,protonet)
        if val_proto_acc > best_acc:
            best_acc = val_proto_acc

        # 早停检查
        early_stopping(val_proto_acc, model)

        # 如果满足早停条件，停止训练
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print("Validating, Epoch: %d, accuracy: %f, best accuracy: %f" % (epoch, val_proto_acc, best_acc))
        writer.add_scalar("val/loss", val_proto_loss, epoch)
        writer.add_scalar("val/Accuracy", val_proto_acc, epoch)
        writer.add_scalar("val/Best_Acc", best_acc, epoch)



    # 加载最好的模型
    model.load_state_dict(torch.load(early_stopping.best_model_path))
    best_test_loss, best_test_acc, best_conf_matrix = test(parser, data_dset, data_sampler, model, protonet)

    # 如果需要加载最后一个模型
    model.load_state_dict(torch.load(early_stopping.last_model_path))
    last_test_loss, last_test_acc, last_conf_matrix = test(parser, data_dset, data_sampler, model, protonet)

    if (best_test_acc > last_test_acc):
        conf_matrix = best_conf_matrix
    else:
        conf_matrix = last_conf_matrix

    return best_test_loss, best_test_acc, last_test_loss, last_test_acc ,conf_matrix

def visualize_confusion_matrix(conf_matrix, class_names, save_path="confusion_matrix_SEED.png"):
    # 归一化每一行的总和，转为百分比
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_percentage = conf_matrix / row_sums * 100

    plt.figure(figsize=(8, 6))

    # 设置字体为 Times New Roman，使图形更美观
    plt.rc('font', family='DejaVu Sans')

    # 绘制热力图，添加字体样式和粗体数字
    sns.heatmap(
        conf_matrix_percentage,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 14,"weight": "bold"},  # 设置单元格内字体大小和粗细
        # cbar_kws={'format': '%.0f%%'}  # 将颜色条标签显示为百分比
    )

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14, rotation=0)

    # 保存图片并关闭绘图窗口
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"混淆矩阵已保存为 {save_path}")

if __name__ == "__main__":
    parser = get_parser().parse_args()

    eeg_session_names = [['1_1.npz', '2_1.npz', '3_1.npz', '4_1.npz', '5_1.npz','6_1.npz','7_1.npz',
                          '8_1.npz'],
                         ['1_2.npz', '2_2.npz', '3_2.npz', '4_2.npz', '5_2.npz','6_2.npz','7_2.npz',
                          '8_2.npz'],
                         ['1_3.npz', '2_3.npz', '3_3.npz', '4_3.npz', '5_3.npz','6_3.npz','7_3.npz',
                          '8_3.npz']]

    eye_session_names = [['1_1', '2_1', '3_1', '4_1', '5_1','6_1','7_1',
                          '8_1'],
                         ['1_2', '2_2', '3_2', '4_2', '5_2','6_2','7_2',
                          '8_2'],
                         ['1_3', '2_3', '3_3', '4_3', '5_3','6_3','7_3',
                          '8_3']]

    # Load data set

    eeg_path = "/disk2/home/yuankang.fu/SEED-Franch/EEG-DE-features/eeg_used_4s" # change the Data path!!!
    eye_path = "/disk2/home/yuankang.fu/SEED-Franch/Eye-tracking-features/eye_tracking_feature"

    session_acc = []
    total_conf_matrix = np.zeros((3, 3))  # 初始化累积混淆矩阵
    for session in range (0,3):
        sum_acc = 0.0
        mean_acc = 0.0
        acc_list = []
        torch.cuda.empty_cache()
        idx = 0
        print('%%%%%%%%%% Session: {} %%%%%%%%%%'.format(session))
        for subject in range(0,8):
            idx += 1
            src_idx = [i for i in range(8) if i != subject]
            tar_idx = [subject]
            print(f"src_idx:{src_idx}")
            print(f"tar_idx:{tar_idx}")
            print('%%%%%%%%%% Target Subject: {} %%%%%%%%%%'.format(subject))

            # Data loader
            source_eeg_sample, source_eye_sample, source_label = load4data(parser,eeg_path, eye_path,eeg_session_names,eye_session_names,session,src_idx,"full")
            train_eeg_sample, train_eye_sample, train_label = load4data(parser,eeg_path, eye_path,eeg_session_names,eye_session_names,session,tar_idx,"train")
            val_eeg_sample, val_eye_sample, val_label = load4data(parser,eeg_path, eye_path,eeg_session_names,eye_session_names,session,tar_idx,"val")
            test_eeg_sample, test_eye_sample, test_label = load4data(parser,eeg_path, eye_path,eeg_session_names,eye_session_names,session,tar_idx,"test")

            source_eeg_dset = torch.utils.data.TensorDataset(source_eeg_sample, source_label)
            source_eye_dset = torch.utils.data.TensorDataset(source_eye_sample, source_label)

            train_eeg_dset = torch.utils.data.TensorDataset(train_eeg_sample, train_label)
            train_eye_dset = torch.utils.data.TensorDataset(train_eye_sample, train_label)

            val_eeg_dset = torch.utils.data.TensorDataset(val_eeg_sample, val_label)
            val_eye_dset = torch.utils.data.TensorDataset(val_eye_sample, val_label)

            test_eeg_dset = torch.utils.data.TensorDataset(test_eeg_sample, test_label)
            test_eye_dset = torch.utils.data.TensorDataset(test_eye_sample, test_label)

            print(f"source_eeg_dset.size:{len(source_eeg_dset)}")
            print(f"train_eeg_dset.size:{len(train_eeg_dset)}")
            # print(f"train_eye_dset.size:{len(train_eye_dset)}")
            print(f"val_eeg_dset.size:{len(val_eeg_dset)}")
            print(f"test_eeg_dset.size:{len(test_eeg_dset)}")


            source_sampler = PrototypicalBatchSampler(source_label,3,parser.num_support_src + parser.num_query_src,parser.iterations)
            train_sampler = PrototypicalBatchSampler(train_label,3,parser.num_support_tgt,parser.iterations)
            val_sampler = PrototypicalBatchSampler(val_label,3, parser.num_query_tgt,parser.iterations)
            test_sampler = PrototypicalBatchSampler(test_label,3,parser.num_query_tgt,parser.iterations)

            data_set = {"source_eeg_dset": source_eeg_dset, "source_eye_dset": source_eye_dset,
                           "train_eeg_dset": train_eeg_dset, "train_eye_dset": train_eye_dset,
                        "val_eeg_dset": val_eeg_dset, "val_eye_dset": val_eye_dset,
                        "test_eeg_dset": test_eeg_dset, "test_eye_dset": test_eye_dset}

            data_sampler = {"source_sampler": source_sampler, "train_sampler":  train_sampler,
                            "val_sampler":  val_sampler, "test_sampler":  test_sampler}

            # Start
            writer = SummaryWriter("data/tensorboard/experiment/session"+str(session)+"CAMP/"+ "target" + str(subject))
            # 创建早停实例
            early_stopping = EarlyStoppingAccuracy(patience=parser.patience, verbose=True, individual_id=str(subject),
                                                       session_id=str(session))
            best_test_loss, best_test_acc, last_test_loss, last_test_acc, conf_matrix = main(parser, data_set, data_sampler, writer,early_stopping)
            total_conf_matrix += conf_matrix
            print(f"\nsubject :{subject}\tbest_test_acc: {best_test_acc}\tlast_test_acc:{last_test_acc}")
            sum_acc += max(best_test_acc,last_test_acc)
            mean_acc = sum_acc / (idx)
            acc_list.append(max(best_test_acc,last_test_acc).item())
            print(f"\nsubject :{subject}\tmean_acc: {mean_acc}")
            print(f"subject :{subject}\tacc_list:{acc_list}\n")
            writer.close()
        print(f"session:{session}\tmean_acc :{sum_acc / idx}")
        print(f"acc_list:{acc_list}\n")
        session_acc.append((sum_acc / idx).item())
        print(f"session_acc:{session_acc}")
    # visualize_confusion_matrix(total_conf_matrix, class_names=['Negative', 'Neutral', 'Positive'],
    #                            save_path="confusion_matrix_SEED_FRANCH.png")
    print(f"session_acc:{session_acc}")
    print(f"final_mean_acc :{(session_acc[0] + session_acc[1] + session_acc[2]) / 3}")