from keras.models import load_model, Model
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve, auc, classification_report
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap
import seaborn as sns
from keras import metrics
from progressbar import ProgressBar
from processing.train_utils import imagenet_preprocess_input, wp_preprocess_input
from processing.read_images import count_files
from os import listdir
from os.path import join
import numpy as np
from bokeh.plotting import figure, show, output_file
import pandas as pd
import argparse
from processing.clean_csv import create_dir
from sklearn.preprocessing import label_binarize
import os


MODEL_PATH = "models/resnet50_1-24-13-58_empty_tune-3-no-0/retrain-tune-3/19-0.343._retrain_layers-3-s-1/12-0.408.hdf5"
    #IMG_PATH = "../data/wikipaintings_full/wikipaintings_test/Baroque/adriaen-brouwer_village-barbershop.jpg"
    #get_act_map(MODEL_PATH, IMG_PATH, target_size=(224, 224), layer_no=100, plot_size=(8, 8))

DEFAULT_MODEL_PATH = MODEL_PATH
DEFAULT_SAVE_PATH = "models/eval"
N_CLASSES = 25

parser = argparse.ArgumentParser(description='Description')

# parser.add_argument('-t', action="store", default='acc', dest='type', help='Type of Evaluation [acc-predictive accuracy of model]')
parser.add_argument('-m', action="store", dest='model_path', default=DEFAULT_MODEL_PATH, help='Path of the model file')
parser.add_argument('-d', action="store", dest="data_path",
                    default="../data/wikipaintings_small/wikipaintings_test", help="Path of test data")
parser.add_argument('-k', action='store', dest='top_k', default='1,3,5', help='Top-k accuracy to compute')
parser.add_argument('-cm', action="store_true", dest='get_cm', default=False, help='Get Confusion Matrix')
parser.add_argument('-pr', action="store_true", dest='get_pr', default=False, help='Get Precision Recall Curve')
parser.add_argument('--report', action="store_true", dest='get_class_report', default=False,
                    help='Get Classification Report')
parser.add_argument('-show', action="store_true", dest='show_g', default=False, help='Display graphs')
parser.add_argument('-save', action="store", default=DEFAULT_SAVE_PATH, dest='save_path',
                    help='Save graphs, give save location')
# parser.add_argument('--dropout', action="store", default=0.0, type=float, dest='add_drop', help='Add dropout rate [0-1]')
# parser.add_argument('--mom', action="store", default=0.0, type=float, dest='add_mom', help='Add momentum to SGD')
# parser.add_argument('-ln', action="store", type=int, dest='layer_no', help='Select the layer to replace')
args = parser.parse_args()



def evaluate():
    MODEL_PATH = args.model_path
    DATA_PATH = args.data_path
    k = (str(args.top_k)).split(",")
    K= [int(val) for val in k]
    y_pred, y_true = get_acc(MODEL_PATH, DATA_PATH, k=K)
    name = MODEL_PATH.rsplit('/', -1)[1]
    SHOW = args.show_g
    if args.save_path is None:
        SAVE = False
        SAVE_PATH = None
    else:
        SAVE = True
        SAVE_PATH = args.save_path
        create_dir(join(SAVE_PATH, name))
    if args.get_cm is not None:
        get_confusion_matrix(y_true, y_pred, show=SHOW, save=SAVE, path=SAVE_PATH, name=name)
    if args.get_pr is not None:
        get_precision_recall(y_true, y_pred, name, SHOW)
    if args.get_class_report is not None:
        get_classification_report(y_true, y_pred, name, 'classification report: ' + name, show=SHOW, save=SAVE, path=SAVE_PATH)


evaluate()
# https://www.kaggle.com/amarjeet007/visualize-cnn-with-keras


def get_act_map(model_path, img_path, target_size, layer_no, plot_size=(8, 8)):
    # target size is the size of the image to load in
    # given layer 1 (index = 0) is input layer, put any value from 1 onwards in layer_no
    model = load_model(model_path)
    outputs = [layer.output for layer in model.layers]
    print("number of layers", len(model.layers))
    act_model = Model(inputs=model.input, outputs=outputs)
    img = load_img(img_path, target_size=target_size)
    x = img_to_array(img)
    print("shape of x ", x.shape)
    if 'vgg16' or 'resnet50' in model_path:
        x = imagenet_preprocess_input(x)
    else:
        x = wp_preprocess_input(x)
    print("shape of x ", x.shape)
    activations = act_model.predict(x[np.newaxis,:,:,:])
    act = activations[layer_no]
    i = 0
    fig, ax = plt.subplot(plot_size[0], plot_size[1], figsize=(plot_size[0]*2.5, plot_size[1]*1.5))
    for r in range(0, plot_size[0]):
        for c in range(0, plot_size[1]):
            ax[r][c].imshow(act[0, :, :, i])
            i += 1


def get_y_prediction(model_path, test_path, top_k=1, target_size=(224, 224)):
    model = load_model(model_path)
    model.metrics.append(metrics.top_k_categorical_accuracy)
    dico = get_dico()
    y_true = []
    y_pred = []
    s = count_files(test_path)
    styles = listdir(test_path)
    print('Calculating predictions...')
    bar = ProgressBar(max_value=s)
    i = 0
    for style in styles:
        path = join(test_path, style)
        label = dico.get(style)
        imgs = listdir(path)
        for name in imgs:
            img = load_img(join(path, name), target_size=target_size)
            x = img_to_array(img)
            x = wp_preprocess_input(x)
            pred = model.predict(x[np.newaxis, ...])
            args_sorted = np.argsort(pred)[0][::-1]
            y_true.append(label)
            y_pred.append([a for a in args_sorted[:top_k]])
            i += 1
            bar.update(i)
    return y_true, np.asarray(y_pred)

def get_acc(model_path, test_path, k = [1,3,5], target_size=(224, 224)):
    y_true, y_pred = get_y_prediction(model_path, test_path, k[2], target_size)
    acc = []
    one_true = None
    one_pred = None
    for n in k:
        max = len(y_true)
        out = 0
        if n == 1:
            one_true = y_true
            one_pred = y_pred[:, :n]
        for preds, v in zip(y_pred[:,:n], y_true):
            if v in preds:
               out += 1
        acc.append(out/max)
    for n, a in zip(k, acc):
        print("Top-{} accuracy: {}%".format(n,a*100) )
    return one_pred, one_true




def get_confusion_matrix(y_true, y_pred, show=False, normalise=True, save=True, **kwargs):  # output of get_y_prediction
    dico = get_dico()
    cm = confusion_matrix(y_true, y_pred)

    if show:
        plt.figure(figsize=(16, 14))
    if normalise:
       cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
       fmt = '.1f'
            #sns.heatmap(cm, annot=True, fmt=".1f", cmap='Blues', linewidths=.5)
    else:
        fmt = 'd'
            #sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', linewidths=.5)



    '''new_conf_arr = []
    for row in cm:
        new_conf_arr.append(row / sum(row))

    plt.matshow(new_conf_arr)
    plt.yticks(range(25), dico.keys())
    plt.xticks(range(25), dico.keys(), rotation=90)
    plt.colorbar()'''
    plt.title("Confusion matrix")
    classNames = [str(x) for x in list(dico.keys())]
    cm = pd.DataFrame(cm, columns=classNames, index=classNames)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save:
        if 'path' in kwargs:
            path = kwargs['path']
        if 'name' in kwargs:
            name = kwargs['name']
        plt.savefig(path + '/conf_matrix_' + name + '.svg', format='svg')
    if show:
        plt.show()
    plt.show()


    return cm


def get_rates(y_true, y_pred, cm):
    rates = dict()
    row_sum = cm.sum(axis=1)
    len = cm.shape[0]
    total_sum = cm.sum()
    ver_sum = cm.sum(axis=0)

    for i in range(len):
        output = []
        tpr = cm[i][i]/row_sum[i]
        fn = row_sum[i]-cm[i][i]
        fp = ver_sum[i]-cm[i][i]
        tnr = (total_sum-cm[i][i]-fn-fp)/(total_sum-row_sum[i])
        # get classification accuracy and mis-classification
        accuracy = accuracy_score(y_true, y_pred)
        output.append(('accuracy', round(accuracy, 3)))
        output.append(('error', round(1-accuracy, 3)))
        output.append(('sensitivity', round(tpr, 3)))
        output.append(('specificity', round(tnr, 3)))
        rates[i] = output
    return rates


def get_pred(model_path, image_path, top_k=1):
    model = load_model(model_path)
    target_size = (224, 224)
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = wp_preprocess_input(x)
    pred = model.predict(x[np.newaxis, ...])
    dico = get_dico()
    inv_dico = invert_dico(dico)
    args_sorted = np.argsort(pred)[0][::-1]
    preds = [inv_dico.get(a) for a in args_sorted[:top_k]]
    pcts = [pred[0][a] for a in args_sorted[:top_k]]
    return preds, pcts


def get_precision_recall(y_true, y_pred, name, display_all=False):
    values = []
    t_bin = []
    classes = list(range(N_CLASSES))
    print("classes ", classes)
    for i in range(N_CLASSES):
        print(label_binarize(y_true[i], classes=classes))
        #t_bin.append(label_binarize(y_true[i], classes=classes))

    for (t, p) in zip(t_bin, y_pred):
        precision, recall, _ = precision_recall_curve(t.ravel(), p.ravel())
        values.append((precision, recall))
    all_true = np.concatenate(t_bin)
    all_pred = np.concatenate(y_pred)
    all_precision, all_recall, _ = precision_recall_curve(all_true.ravel(), all_pred.ravel())
    area = round(auc(all_recall, all_precision), 2)
    if not display_all:
        x = np.linspace(0.1, 5, 80)

        p = figure(title='Precision & Recall for ' + name, y_axis_type=float,
                   x_range=(0.0, max(all_recall)+1.0), y_range=(0.0, max(all_precision)+1.0),
                   background_fill_color="#fafafa")

        p.line(all_recall, all_precision, legend='Overall AUC %.4f' % (round(auc(all_recall, all_precision, 2))),
               line_color="tomato", line_dash="dashed")
        p.legend.location = "top_left"
        file_name = 'pr_curve_'+name
        output_file(file_name+".html", title=file_name)
        show(p)

    return area

    '''old: show pr curve 
    showPRCurve(values, allPrecision, allRecall, type, name, path, show):
    plt.figure(figsize=[6, 4.5])
    for i, (precision, recall) in enumerate(values):
        label = 'Fold %d AUC %.4f' % (i + 1, auc(recall, precision))
        plt.step(recall, precision, label=label)
    label = 'Overall AUC %.4f' % (auc(allRecall, allPrecision))
    plt.step(allRecall, allPrecision, label=label, lw=2, color='black')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('{0} - PR Curve'.format(name))
    plt.legend(loc='lower left', fontsize='small')
    plt.savefig(path + '/' + type + '_prc_' + name + '.svg', format = "svg")
    if show:
        plt.show()
    '''

# using bokeh to display interactive confusion matrix (possible to hover and save)


def display_cm_hover(model_path, y_true, y_pred):
    name = model_path.split('/')[-2] + '-' + model_path.rsplit('/', 1)[1].replace('hdf5', '')
    cm = get_confusion_matrix(y_true, y_pred)
    names = list(get_dico().keys())
    N = len(names)
    counts = np.zeros((N, N))
    total = 0
    for i in range(N):
        for j in range(N):
            counts[i, j] = cm[i][j]
            total += counts[i, j]
    colormap = ["#444444", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
                "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"]

    xname = []
    yname = []
    color = []
    alpha = []
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(list(reversed(names))):
            xname.append(name1)
            yname.append(name2)

            alpha.append(counts[i, j]/cm.sum(axis=1)[i])
            color.append("#79145a")

    data = dict(
        xname=xname,
        yname=yname,
        colors=color,
        alphas=alpha,
        count=counts.flatten(),
    )

    p = figure(title="Confusion Matrix of Model " + name,
               x_axis_location="above", tools="hover,save",
               x_range=names, y_range=list(reversed(names)),
               tooltips=[('True', '@yname'), ('Predicted' ,'@xname'), ('Count', '@count')])

    p.plot_width = 800
    p.plot_height = 800
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi / 3

    p.rect('xname', 'yname', 0.9, 0.9, source=data,
           color='colors', alpha='alphas', line_color=None,
           hover_line_color='black', hover_color='colors')
    file_name = 'conf_' + name
    output_file(file_name + '.html', title=file_name)

    show(p)  # show the plot

def get_classification_report(y_true, y_pred, name, title, show=True, save=False, **kwargs):
    ddl_heat = ['#DBDBDB', '#DCD5CC', '#DCCEBE', '#DDC8AF', '#DEC2A0', '#DEBB91',
                '#DFB583', '#DFAE74', '#E0A865', '#E1A256', '#E19B48', '#E29539']
    ddlheatmap = colors.ListedColormap(ddl_heat)
    title = title or 'Classification report'
    cr = classification_report(y_true, y_pred, name)
    lines = cr.split('\n')
    classes = []
    matrix = []

    for line in lines[2:(len(lines) - 3)]:
        s = line.split()
        classes.append(s[0])
        value = [float(x) for x in s[1: len(s) - 1]]
        matrix.append(value)

    if show:
        fig, ax = plt.subplots(1)

        for column in range(len(matrix) + 1):
            for row in range(len(classes)):
                txt = matrix[row][column]
                ax.text(column, row, matrix[row][column], va='center', ha='center')

        fig = plt.imshow(matrix, interpolation='nearest', cmap=ddlheatmap)
        plt.title(title)
        plt.colorbar()
        x_tick_marks = np.arange(len(classes) + 1)
        y_tick_marks = np.arange(len(classes))
        plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
        plt.yticks(y_tick_marks, classes)
        plt.ylabel('Classes')
        plt.xlabel('Measures')
        plt.show()
    if save:
        if 'path' in kwargs:
            path = kwargs['path']
        plt.savefig(path + '/conf_matrix_' + name + '.svg', format='svg')
    # from https://medium.com/district-data-labs/visual-diagnostics-for-more-informed-machine-learning-7ec92960c96b

def plot_history(history):
    loss_list = history['loss']
    val_loss_list = history['val_loss']
    acc_list = history['acc']
    val_acc_list = history['val_acc']

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return
        ## As loss always exists
    epochs = range(1, len(loss_list) + 1)

    ## Loss
    plt.figure(1)
    plt.plot(epochs, loss_list, 'b',
             label='Training loss (' + str(str(format(loss_list[-1], '.5f')) + ')'))
    plt.plot(epochs, val_loss_list, 'g',
             label='Validation loss (' + str(str(format(val_loss_list[-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## Accuracy
    plt.figure(2)
    plt.plot(epochs, acc_list, 'b',label='Training accuracy (' + str(format(acc_list[-1], '.5f')) + ')')
    plt.plot(epochs, val_acc_list, 'g', label='Validation accuracy (' + str(format(val_acc_list[-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def decode_image_autoencoder(model_path, img_path):
    autoencoder = load_model(model_path)
    img = load_img(img_path)
    x = img_to_array(img)
    dec = autoencoder.predict(x)  # Decoded image
    x = x[0]
    dec = dec[0]
    x = (x.transpose((1, 2, 0)) * 255).astype('uint8')
    dec = (dec.transpose((1, 2, 0)) * 255).astype('uint8')

    plt.imshow(np.hstack((x, dec)))
    plt.title('Original and reconstructed images')
    plt.show()

# FUNCTION FROM rasta.python.utils.utils
def get_dico():
    classes = []
    PATH = os.path.dirname(__file__)
    directory = join(PATH,'data/wikipaintings_small/wikipaintings_train')
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)
    class_indices = dict(zip(classes, range(len(classes))))
    return class_indices

def invert_dico(dico):
    return {v: k for k, v in dico.items()}


    #his = pickle.load(open('../models/resnet50_1-24-13-58_empty_tune-3-no-0/retrain-tune-3/19-0.343._retrain_layers-3-s-1/12-0.408._retrain_layers-3-s-4/history.pck', 'rb'))
    # print(his)
    #plot_history(his)
    # resnet50_1-24-13-58_empty_tune-3-no-0/retraintune-7-no-0/04-0.144._tune-7-no-0/06-0.162._tune-8-no-1/04-0.180.hdf5
    #MODEL_PATH = "../models/resnet50_1-24-13-58_empty_tune-3-no-0/retraintune-7-no-0/04-0.144._tune-7-no-0/06-0.162._tune-8-no-1/04-0.180.hdf5"

    #DIR_PATH = "../models/resnet50_1-24-13-58_empty_tune-3-no-0/retrain-tune-3/19-0.343._retrain_layers-3-s-1"
    #y_true, y_pred = get_y_prediction(MODEL_PATH,"../data/wikipaintings_small/wikipaintings_test")
    #get_confusion_matrix(y_true, y_pred, show=True, path=DIR_PATH, name="12-0.408")
    #display_cm_hover(MODEL_PATH, y_true, y_pred)
    #print(get_dico())
    #names = list(get_dico().keys())
    #print(names)
    #print("in format")
    #print(list(reversed(names)))