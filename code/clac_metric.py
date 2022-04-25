import numpy as np
import matplotlib.pyplot as plt
def get_metrics(real_score, predict_score):
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    # 160373
    sorted_predict_score_num = len(sorted_predict_score)
    # 阈值的选取是从 sorted_predict_score中按从下到大的顺序每间隔1000取一个score作为阈值，一共选取了999个threshold(从小到大)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    # predict_score_matrix 是999*160862,每一行可以根据一个threshold计算出混淆矩阵
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    # predict_score_matrix是999*160862,thresholds转置后是999*1,通过广播机制进行比较，那么，999行的每一行都是基于不同的thresholds判断正负样本
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    # predict_score_matrix(999,160862),到这里一行就是根据一个阈值判断的正负样本。
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    # 999*160862 与16082*1的矩阵做矩阵乘法，得到999*1(999个数字每个数字都是TP的数量，且阈值越大，预测对的正例也多)
    TP = predict_score_matrix.dot(real_score.T)
    # FP=预测的正例数量-TP(预测对的正例数量)(阈值越大，预测对正例(TP)的就也多，相应的fp也就越小）
    FP = predict_score_matrix.sum(axis=1) - TP
    # 实际是正例的-预测对的是正例的数量=被误判为负例的正例（FN）
    FN = real_score.sum() - TP
    # 判断对的负例=样本数-判断对正例-误判为正的负例-误判为负的正例
    TN = len(real_score.T) - TP - FP - FN

    # fpr：负例中被误判为正例的比率(低好)(阈值越大，fpr越小,因为被判断为正例的数量就很少了，判断错的也就少了)
    fpr = FP / (FP + TN)
    # tpr：正例中判断为正例的比率(就是recall，高好)(阈值越大，tpr小，因为本身判断为正的数量也少)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    # 为了让roc曲线从原点开始(将第一个坐标代替为(0,0))
    print('ROC_dot_matrix.shape:',ROC_dot_matrix.shape)
    ROC_dot_matrix.T[0] = [0, 0]
    # 为了让roc曲线到（1,1）结束，np.c_：按行连接
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T

    plt.plot(x_ROC,y_ROC)
    # 计算无数个梯形面积来拟合pr曲线下的面积 x_ROC[1:]-x_ROC[:-1]计算的是高(其实就是后一个x减去前一个x),y_ROC[:-1]+y_ROC[1:]是上底加下底(前一个y+后一个y）
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])
    # recall 就是tpr
    recall_list = tpr
    # 预测为正例中预测对的比例（阈值越大，pr也大）
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    # 保证pr曲线从（0,1）开始，1,0 结束
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    plt.plot(x_PR,y_PR)
    plt.show()
    # 计算无数个梯形面积来拟合pr曲线下的面积 x_PR[1:]-x_PR[:-1]计算的是高(其实就是后一个x减去前一个x),y_PR[:-1]+y_PR[1:]是上底加下底(集前一个y+后一个y）
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    # 分类正确的比率
    accuracy_list = (TP + TN) / len(real_score.T)
    # 被分对的负例占负例的比例
    specificity_list = TN / (TN + FP)
    #这里以是的f1-score最大时候所对应的阈值几算的性能指标作为最佳性能指标
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    print( ' auc:{:.4f} ,aupr:{:.4f},f1_score:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, specificity:{:.4f}, precision:{:.4f}'.format(
        auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision))
    return [auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision]
