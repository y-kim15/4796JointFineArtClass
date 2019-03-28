from keras.models import load_model, Model
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from keras import metrics
from progressbar import ProgressBar
from processing.train_utils import imagenet_preprocess_input, wp_preprocess_input, id_preprocess_input
from processing.read_images import count_files
from os import listdir
from os.path import join
import numpy as np
from scipy import interp
from bokeh.plotting import figure, show, output_file
import pandas as pd
import argparse
import os, json
import pickle
import csv
import math

MODEL_PATH = "models/resnet50_2-4-15-35_empty_layers-3-s-0/09-0.487._retrain_layers-172-s-1/13-0.354._retrain_layers-168,172,178-s-2/12-0.375.hdf5"
    #"models/resnet50_1-24-13-58_empty_tune-3-no-0/retrain-tune-3/19-0.343._retrain_layers-3-s-1/12-0.408.hdf5"
IMG_PATH = "data/wikipaintings_full/wikipaintings_test/Baroque/annibale-carracci_triumph-of-bacchus-and-ariadne-1602.jpg"

DEFAULT_MODEL_PATH = MODEL_PATH
DEFAULT_SAVE_PATH = "models/eval"
N_CLASSES = 25

parser = argparse.ArgumentParser(description='Description')

parser.add_argument('-t', action="store", dest='type', help='Type of Evaluation [acc-predictive accuracy of model][acc|pred]')
parser.add_argument('-cv', action="store", dest='cv', help='Evaluate Cross Validation Output and Save [path to csv to save]' )
parser.add_argument('-m', action="store", dest='model_path', default=DEFAULT_MODEL_PATH, help='Path of the model file')
parser.add_argument('-d', action="store", dest="data_path", help="Path of test data")
parser.add_argument('-ds', action="store", dest="data_size", choices=['f', 's'], help="Choose the size of test set, full or small")
parser.add_argument('-dp', action="store_true", dest='lab', default=True, help="Set to test in lab", )
parser.add_argument('-k', action='store', dest='top_k', default='1,3,5', help='Top-k accuracy to compute')
parser.add_argument('-cm', action="store_true", dest='get_cm', default=False, help='Get Confusion Matrix')
parser.add_argument('--report', action="store_true", dest='get_class_report', default=False,
                    help='Get Classification Report')
parser.add_argument('--show', action="store_true", dest='show_g', default=False, help='Display graphs')
parser.add_argument('-s', action="store_true", dest='save', default=False, help='Save graphs')
parser.add_argument('--save', action="store", default=DEFAULT_SAVE_PATH, dest='save_path',
                    help='Specify save location')
parser.add_argument('--his', action="store", dest='plot_his', help='Plot history, choose which to plot [l|a|b (default)]')
parser.add_argument('-f', action="store", dest="file", help='Name of history file to plot (extension pck)')
parser.add_argument('--model_name', action="store", dest='model_name', help='Model types/name: Mandatory to call --his')
parser.add_argument('--act', action="store", dest='act', help='Visualise activation function of layer (layer name or index)')
parser.add_argument('--roc', action="store_true", dest='get_roc', help='Get Roc Curve')
args = parser.parse_args()

# https://www.kaggle.com/amarjeet007/visualize-cnn-with-keras


def get_act_map(model_path, img_path, target_size, layer_no, plot_size=(4,4)):
    # target size is the size of the image to load in
    # given layer 1 (index = 0) is input layer, put any value from 1 onwards in layer_no
    model = load_model(model_path)

    outputs = [layer.output for layer in model.layers]
    outputs = outputs[1:]
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
    new_x = x[np.newaxis,:,:,:]
    print("new shape of x ", x.shape)
    activations = act_model.predict(new_x, batch_size=1)
    if layer_no.isdigit() or isinstance(layer_no, int):
        act = activations[layer_no]
    else: #returned is name of layer
        index = None
        for idx, layer in enumerate(model.layers):
            if layer.name == layer_no:
                index = idx
                print("layer name: ", layer_no, ", with index: ", index)
                break
        act = activations[index]
    i = 0
    print("original")
    plt.imshow(img)
    print("plotsize 0, ", plot_size[0])
    print("plotsize 1, ", plot_size[1])
    #f = plt.figure(figsize=(plot_size[0]*2.5, plot_size[1]*1.5))
    #ax = f.add_subplot(plot_size[0], plot_size[1], pow(plot_size[0],2), )
    f, ax = plt.subplots(plot_size[0], plot_size[1], squeeze=False)
    #fig, ax = plt.subplot(plot_size[0], plot_size[1], pow(plot_size[0],2), figsize=(plot_size[0] * 2.5, plot_size[1] * 1.5))
    for r in range(0, plot_size[0]):
        for c in range(0, plot_size[1]):
            ax[r][c].imshow(act[0, :, :, i])
            i += 1

    plt.show()


def get_y_prediction(model_path, test_path, top_k=1, target_size=(224, 224)):
    model = load_model(model_path)
    model.metrics.append(metrics.top_k_categorical_accuracy)
    dico = get_dico()
    y_true = []
    y_pred = []
    y_pred_k = []
    s = count_files(test_path)
    styles = listdir(test_path)
    print('Calculating predictions...')
    bar = ProgressBar(max_value=s)
    i = 0
    for style in styles:
        path = join(test_path, style)
        label = dico.get(style)
        imgs = listdir(path)
        y_t = []
        y_p = []
        for name in imgs:
            img = load_img(join(path, name), target_size=target_size)
            x = img_to_array(img)
            x = wp_preprocess_input(x)
            pred = model.predict(x[np.newaxis, ...])
            args_sorted = np.argsort(pred)[0][::-1]
            y_t.append(label)
            y_p.append([a for a in args_sorted[:top_k]])
            y_pred_k.append([a for a in args_sorted[:top_k]])
            i += 1
            bar.update(i)
        y_true.append(y_t)
        y_pred.append(y_p)
    return y_true, y_pred, np.asarray(y_pred_k) #np.asarray(y_pred)


def get_acc(model_path, test_path, k=[1, 3, 5], target_size=(224, 224)):
    y_t, _ ,y_pred = get_y_prediction(model_path, test_path, k[2], target_size)
    y_true = [j for sub in y_t for j in sub]
    #y_pred = [j for sub in y_p for j in sub]
    acc = []
    one_true = None
    one_pred = None
    for n in k:
        max = len(y_true)
        out = 0
        if n == 1:
            one_true = y_true
            one_pred = y_pred[:, :n]
        for preds, v in zip(y_pred[:, :n], y_true):
            if v in preds:
                out += 1
        acc.append(out / max)
    for n, a in zip(k, acc):
        print("Top-{} accuracy: {}%".format(n, a * 100))
    y_p = [x[0] for x in y_pred]
    return y_t, y_p, one_true, one_pred, k, acc


def get_confusion_matrix(y_true, y_pred, show, normalise=True, save=False, **kwargs):  # output of get_y_prediction
    dico = get_dico()
    cm = confusion_matrix(y_true, y_pred)
    orig_cm = cm
    if show:
        plt.figure(figsize=(18, 18))
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        # sns.heatmap(cm, annot=True, fmt=".1f", cmap='Blues', linewidths=.5)
    else:
        fmt = 'd'
        # sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', linewidths=.5)

    plt.title("Confusion matrix")
    classNames = [str(x) for x in list(dico.keys())]
    cm = pd.DataFrame(cm, columns=classNames, index=classNames)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    path = None
    name = None
    if save:
        if 'path' in kwargs:
            path = kwargs['path']
        if 'name' in kwargs:
            name = kwargs['name']
        plt.savefig(path + '/conf_matrix_' + name + '.svg', format='svg')
    if show:
        plt.show()
    print(cm)
    return orig_cm

# Finds the optimal cut-off point which returns the optimal true positive rate with
# minimal false positive rate. This is then plotted on graph.
def find_opt_threshold(fprs, tprs):
    current = (fprs[0], tprs[0])
    distance = -1
    for i in range(len(fprs)):
        dis = math.sqrt(((1-tprs[i])**2 + fprs[i]**2))
        if distance == -1 or (distance > dis and dis > 0) :
            distance = dis
            current = (fprs[i], tprs[i])
    return current


def get_roc_curve(y_true, y_pred, save, path, show=False, **kwargs):
    k = 1
    nSamples = len(y_true)
    dico = get_dico()
    classes = [str(x) for x in list(dico.keys())]
    # Computes ROC curve and area for each class
    rocAuc = dict()
    fpr = dict()
    tpr = dict()
    #print("shape of y_true ", str(y_true.shape), " shape of y_pred ", str(y_pred.shape))
    for i in range(len(classes)):
        tprs = []
        meanFpr = np.linspace(0, 1, nSamples)
        # Computes by k fold cross validation where length of self.values_ denotes value of k
        jFpr, jTpr, _ = roc_curve(y_true[:,i], y_pred[:,i])
        tprs.append(interp(meanFpr, jFpr, jTpr))
        tprs[-1][0] = 0.
        # for j in range(k):
        #     jFpr, jTpr, _ = roc_curve(y_true[j][:, i], y_pred[j][:, i])
        #     tprs.append(interp(meanFpr, jFpr, jTpr))
        #     tprs[-1][0] = 0.

        meanTpr = np.mean(tprs, axis=0)
        meanTpr[-1] = 1.0
        rocAuc[i] = dict()
        rocAuc[i]["area"] = round(auc(meanFpr, meanTpr), 2)
        fpr[i] = meanFpr
        tpr[i] = meanTpr
        optFpr, optTpr = find_opt_threshold(fpr[i], tpr[i])
        rocAuc[i]["optimal"] =  [("fpr", round(optFpr, 2)), ("tpr", round(optTpr, 2))]
    find_micro(y_true, y_pred, k, nSamples, fpr, tpr, rocAuc)
    find_macro(fpr, tpr, rocAuc, classes)

    name = None
    if save:
        name = kwargs['name']
    show_roc_curve(fpr, tpr, rocAuc, classes, name, path, show=show)
    return rocAuc

def find_micro(y_true, y_pred, k, nSamples, fpr, tpr, rocAuc):
    # Computes micro-average ROC curve and area
    tprsMicro = []
    meanFpr = np.linspace(0, 1, nSamples)
    fprMicro, tprMicro, _ = roc_curve(y_true, y_pred)
    tprsMicro.append(interp(meanFpr, fprMicro, tprMicro))
    tprsMicro[-1][0] = 0.
    # for j in range(k):
    #     fprMicro, tprMicro, _ = roc_curve(y_true[j].ravel(), y_pred[j].ravel())
    #     tprsMicro.append(interp(meanFpr, fprMicro, tprMicro))
    #     tprsMicro[-1][0] = 0.
    meanTprMicro = np.mean(tprsMicro, axis=0)
    meanTprMicro[-1] = 1.0
    fpr["micro"] = meanFpr
    tpr["micro"] = meanTprMicro
    rocAuc["micro"] = round(auc(meanFpr, meanTprMicro), 2)

def find_macro(fpr, tpr, rocAuc, classes):
    # Computes macro-average ROC curve and area
    allFpr = np.unique(np.concatenate([fpr[i] for i in range(classes.shape[0])]))
    meanTpr = np.zeros_like(allFpr)
    for i in range(classes.shape[0]):
        meanTpr += interp(allFpr, fpr[i], tpr[i])
    meanTpr /= classes.shape[0]

    fpr["macro"] = allFpr
    tpr["macro"] = meanTpr
    rocAuc["macro"] = round(auc(fpr["macro"], tpr["macro"]), 2)


def show_roc_curve(fpr, tpr, rocAuc, classes, name, path, show=False):
    fig = plt.figure(figsize=[6, 4.5])
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.3f})'
                   ''.format(rocAuc["micro"]), color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(rocAuc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    '''
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'lawngreen', 'purple'
    'dimgrey', 'gold', 'darkcyan', 'crimson', 'darkgreen'])
    for i, color in zip(range(classes.shape[0]), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, rocAuc[i]["area"]))'''
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey',
            alpha=.2)
    plot_roc_grid('multi', name, path, fig, show=show)

def plot_roc_grid(type, name, path, fig, show=False):
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('{0} - ROC Curve'.format(name))
    plt.legend(loc='lower right', fontsize='small')
    if name is not None:
        plt.savefig(path + '/' + type + '_roc_' + name + '.svg', format = "svg")

    if show:
        if name==None:
            name = 'roc_curve_example'
        plt.show(fig)
    else:
        plt.close(fig)


def get_rates(y_true, y_pred, cm):
    rates = dict()
    row_sum = cm.sum(axis=1)
    len = cm.shape[0]
    total_sum = cm.sum()
    ver_sum = cm.sum(axis=0)

    for i in range(len):
        output = []
        tpr = cm[i][i] / row_sum[i]
        fn = row_sum[i] - cm[i][i]
        fp = ver_sum[i] - cm[i][i]
        tnr = (total_sum - cm[i][i] - fn - fp) / (total_sum - row_sum[i])
        # get classification accuracy and mis-classification
        accuracy = accuracy_score(y_true, y_pred)
        output.append(('accuracy', round(accuracy, 3)))
        output.append(('error', round(1 - accuracy, 3)))
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
    #print("predicted percents: ")
    #print(args_sorted[:top_k])
    preds = [inv_dico.get(a) for a in args_sorted[:top_k]]
    pcts = [pred[0][a] for a in args_sorted[:top_k]]
    return preds, pcts


def get_classification_report(y_true, y_pred, labels, name, title, show=True, save=False, **kwargs):
    ddl_heat = ['#DBDBDB', '#DCD5CC', '#DCCEBE', '#DDC8AF', '#DEC2A0', '#DEBB91',
                '#DFB583', '#DFAE74', '#E0A865', '#E1A256', '#E19B48', '#E29539']
    ddlheatmap = colors.ListedColormap(ddl_heat)
    title = title or 'Classification report'
    cr = classification_report(y_true, y_pred, target_names=labels)
    print(cr)
    lines = cr.split('\n')
    classes = []
    matrix = []
    print(lines)
    print("get lines")
    for line in lines[2:(len(lines) - 5)]:
        print(line)
        s = line.split()
        print("s: ", s)
        classes.append(s[0])
        value = [float(x) for x in s[1: len(s) - 1]]
        matrix.append(value)

    plt.figure(figsize=(20, 16))
    # fig, ax = plt.subplots(1, squeeze=False)
    print("len of matrix: ", len(matrix))
    print("len of classes: ", len(classes))
    matrix = pd.DataFrame(matrix, columns=['precision', 'recall', 'f1-score'], index=classes)
    ax = sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues')

    ax.figure.tight_layout()
    #ax.figure.subplots_adjust(left=0.5)
    plt.title(title)
    plt.ylabel('Classes')
    plt.xlabel('Measures')

    if save:
        if 'path' in kwargs:
            path = kwargs['path']
        plt.savefig(path + '/class_report_' + name + '.svg', format='svg', bbox_inches='tight')

    if show:

        plt.show()


    # from https://medium.com/district-data-labs/visual-diagnostics-for-more-informed-machine-learning-7ec92960c96b

def plot_history_hover(history, type, add_title=True, show_up=True, save=False, **kwargs):
    from collections import defaultdict

    import numpy as np
    from scipy.stats import norm

    from bokeh.plotting import show, figure
    from bokeh.models import ColumnDataSource, HoverTool, Title
    from bokeh.palettes import Viridis6
    from bokeh.models.tools import CustomJSHover
    if type == 'Loss':
        list1 = ['%.3f' % elem for elem in history['loss']]
        list2 = [ '%.3f' % elem for elem in history['val_loss']]
    else:
        list1 = [ '%.3f' % elem for elem in history['acc']]
        list2 = [ '%.3f' % elem for elem in history['val_acc']]
    if len(list1) == 0 and type == 'Loss':
        print('Loss is missing in history')
        return
    else:
        print("length: ", len(list1))
        ## As loss always exists

    epochs = np.array(list(range(1, len(list1) + 1)))
    print(epochs)
    if save:
        if 'path' in kwargs:
            path = kwargs['path']
        if 'name' in kwargs:
            name = kwargs['name']
        output_file(join(path, name+'.html'))

    data = defaultdict(list)
    for t, val in zip(['Train', 'Val'], [list1, list2]):#[(1.0, 83), (0.9, 55), (0.6, 98), (0.4, 43), (0.2, 39), (0.12, 29)]:
        data["Epochs"].append(epochs)
        data[type].append(val)
        data['Type'].append(t)
    data['color'] = Viridis6

    source = ColumnDataSource(data)

    p = figure(plot_height=400, x_axis_label='Epochs', y_axis_label=type)
    p.multi_line(xs='Epochs', ys=type, legend="Type",
                 line_width=5, line_color='color', line_alpha=0.6,
                 hover_line_color='color', hover_line_alpha=1.0,
                 source=source)

    '''p.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[
        ('Loss', '@Loss')
    ]))'''
    e = CustomJSHover(code="""
        return '' + special_vars.data_x
    """)

    l = CustomJSHover(code="""
        return '' + special_vars.data_y
    """)

    p.add_tools(
        HoverTool(
            show_arrow=False,
            line_policy='next',
            tooltips=[
                ('Epochs', '@Epochs{custom}'),  # or just ('X_value', '$data_x')
                (type, '@'+type+'{custom}')
            ],
            formatters={'Epochs': e, type:l}
        )
    )
    if add_title:
        p.add_layout(Title(text='Training History Plot', text_font_style="italic"), 'above')
        p.add_layout(Title(text=name, text_font_size="16pt"), 'above')

    if show_up:
        show(p)

#https://plot.ly/python/line-charts/
def plot_history_plotly(history, type, name, path=None, save=False, show=False):
    import plotly
    import plotly.graph_objs as go
    if type == 'Loss':
        d1 = history['loss']
        d2 = history['val_loss']
    else:
        d1 = history['acc']
        d2 = history['val_acc']

    if type == 'Loss' and len(d1) == 0:
        print('Loss is missing in history')
        return
        ## As loss always exists
    epochs = list(range(1, len(d1) + 1))

    # Create and style traces
    trace0 = go.Scatter(
        x=epochs,
        y=d1,
        name='Train ' + type,
        line=dict(
            color='#440154', width=4)
    )
    # '#440154', '#404387', '#29788E', '#22A784', '#79D151', '#FDE724
    trace1 = go.Scatter(
        x=epochs,
        y=d2,
        name='Val ' + type,
        line=dict(
            color='#ff8080', width=4)
    )

    data = [trace0, trace1]

    # Edit the layout
    layout = dict(title='Training History Plot',
                  xaxis=dict(title='Epochs'),
                  yaxis=dict(title=type),
                  )

    fig = dict(data=data, layout=layout)

    if save:
        open = False
        if show:
            open = True
        plotly.offline.plot(fig, filename=join(path, name+'.html'), auto_open=open)
        print(join(path, name + '.html'))
        print("saved")

def show_image(img_path):
    img = load_img(img_path, target_size=(224,224))
    x = img_to_array(img)
    plt.imshow(x/255.)
    plt.title('Original image')
    plt.savefig(img_path.rsplit('/', 1)[1] + '_scaled.jpg')
    plt.show()

def decode_image_autoencoder(model_path, img_path, target_size):
    autoencoder = load_model(model_path)
    img = load_img(img_path, target_size=target_size)
    x = img_to_array(img)
    orig = np.array(x, copy=True)
    y = id_preprocess_input(x)
    y = y[np.newaxis,:,:,:]
    #orig = orig[np.newaxis,:,:,:]
    dec = autoencoder.predict(y)  # Decoded image
    orig = orig/255.
    dec = dec[0][:, :, ::-1]
    dec = dec/255.
    #x = (x.transpose((1, 2, 0)) ).astype('uint8') #removed *255 inside as.type lhs
    #dec = (dec.transpose((1, 2, 0)) ).astype('uint8')

    plt.imshow(np.hstack((orig, dec)))
    plt.title('Original and reconstructed images')
    plt.show()


# FUNCTION FROM rasta.python.utils.utils
def get_dico():
    classes = []
    PATH = os.path.dirname(__file__)
    #directory = join(PATH, 'data/wikipaintings_small/wikipaintings_train')
    directory = "/cs/tmp/yk30/data/wikipaintings_small/wikipaintings_train"
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)
    class_indices = dict(zip(classes, range(len(classes))))
    return class_indices


def invert_dico(dico):
    return {v: k for k, v in dico.items()}

def write_to_csv(path, data, type):
    FIELDNAMES = list(data.keys()).sort()
    with open(join(path, '_'+type+'.csv'), 'a+', newline='') as f:
        lines = f.readlines()
        l = len(lines)
        head = True
        if l > 1:
            head = False
        w = csv.DictWriter(f, FIELDNAMES)
        if head:
            w.writeheader()
        w.writerow(row for row in zip(*(data[key] for key in FIELDNAMES)))


def evaluate():
    MODEL_PATH = args.model_path
    PRE_PATH = ''
    if args.lab:
        PRE_PATH = '/cs/tmp/yk30/'
    if args.data_path is not None:
        DATA_PATH = args.data_path
    else:
        if args.data_size is not None and args.data_size=='f':
            DATA_PATH = PRE_PATH + "data/wikipaintings_full/wikipaintings_test"
        else:
            DATA_PATH = PRE_PATH + "data/wikipaintings_small/wikipaintings_test"
    k = (str(args.top_k)).split(",")
    K = [int(val) for val in k]
    SAVE = args.save
    if SAVE:
        SAVE_PATH = args.save_path
    # SAVE_PATH = join(args.save_path, name)
    # create_dir(join(SAVE_PATH, name))
    else:
        SAVE_PATH = None
    if args.type == 'acc':
        y_t, y_p, y_true, y_pred, k, acc = get_acc(MODEL_PATH, DATA_PATH, k=K)
        name = MODEL_PATH.rsplit('/', 1)[1].replace('.hdf5', '')
        SHOW = args.show_g


        if args.get_cm:
            cm = get_confusion_matrix(y_true, y_pred, show=SHOW, save=SAVE, path=SAVE_PATH, name=name)
        if args.get_class_report:
            classes = get_dico().keys()
            get_classification_report(y_true, y_pred, classes, name, 'classification report: ' + name, show=SHOW,
                                      save=SAVE,
                                      path=SAVE_PATH)
        if args.get_roc:
            get_roc_curve(y_t, y_p, show=SHOW, save=SAVE, path=SAVE_PATH, name=name)
        if args.cv:
            csv_path = args.cv
            dic = {(key, value) for (key, value) in zip(k, acc)}
            write_to_csv(csv_path, dic,'cv')


    else:
        v = 5#K[0]
        pred, pcts = get_pred(MODEL_PATH, DATA_PATH, top_k=v)
        print(pcts)
        if SAVE:
            result = {'pred': pred, 'k': v}
            print(json.dumps(result))
        else:
            print("Top-{} prediction : {}".format(k, pred))
his_type = {
    'a': 'Accuracy',
    'l': 'Loss'
}

def plot():
    if args.plot_his is not None:
        his_t = args.plot_his
        his = pickle.load(open(args.file, 'rb'))
        if his_t == 'b':
            his_t = 'a'
        plot_history_plotly(his, his_type[his_t], save=args.save, path=args.save_path, name=args.model_name+" "+his_type[his_t]+" Plot", show=args.show_g)
        if args.plot_his == 'b':
            his_t = 'l'
            plot_history_plotly(his, his_type[his_t], save=args.save, path=args.save_path, name=args.model_name + " " + his_type[his_t] + " Plot", show=args.show_g)
    elif args.act is not None:
        MODEL_PATH = args.model_path
        DATA_PATH = args.data_path
        #model_path, img_path, target_size, layer_no, plot_size=(2,2))
        layer_no = args.act
        get_act_map(MODEL_PATH, DATA_PATH, (224, 224), layer_no=layer_no)

# evaluate for finding model accuracy and prediction by running the model on test set
# should have -t tag indicating one of the options
if args.type is not None:
    evaluate()
# anything else including visualising activation maps and plotting history
# --act (need -m, -d (has to be a path to an image file in this case), --act (layer name/index), --show, -s, --save)
# --his (need -f, --show, -s, --save)
else:
    plot()

#show_image(IMG_PATH)
#decode_image_autoencoder(MODEL_PATH, IMG_PATH, (224,224))
#get_act_map(MODEL_PATH, IMG_PATH, target_size=(224, 224), layer_no=171)

#his = pickle.load(open('models/resnet50_2-4-15-35_empty_layers-3-s-0/history.pck', 'rb'))
    #'models/resnet50_1-24-13-58_empty_tune-3-no-0/retrain-tune-3/19-0.343._retrain_layers-3-s-1/'
     #                  '12-0.408._retrain_layers-3-s-4/history.pck', 'rb'))
#print(his)
#plot_history_plotly(his, 'Accuracy', save=True, path='models/eval/resnet50_1-24-13-58_empty_tune-3-no-0', name='resnet50_retrained_12-0.40_acc_plotly')
#plot_history_plotly(his, 'Loss', save=True, path='models/eval/resnet50_1-24-13-58_empty_tune-3-no-0', name='resnet50_retrained_12-0.40_loss_plotly')
#plot_history_hover(his, 'Loss', save=True, path='models/eval/resnet50_1-24-13-58_empty_tune-3-no-0', name='resnet50_retrained_12-0.40_loss')
    # resnet50_1-24-13-58_empty_tune-3-no-0/retraintune-7-no-0/04-0.144._tune-7-no-0/06-0.162._tune-8-no-1/04-0.180.hdf5
    # MODEL_PATH = "../models/resnet50_1-24-13-58_empty_tune-3-no-0/retraintune-7-no-0/04-0.144._tune-7-no-0/06-0.162._tune-8-no-1/04-0.180.hdf5"

    # DIR_PATH = "../models/resnet50_1-24-13-58_empty_tune-3-no-0/retrain-tune-3/19-0.343._retrain_layers-3-s-1"
    # y_true, y_pred = get_y_prediction(MODEL_PATH,"../data/wikipaintings_small/wikipaintings_test")
    # get_confusion_matrix(y_true, y_pred, show=True, path=DIR_PATH, name="12-0.408")
    # display_cm_hover(MODEL_PATH, y_true, y_pred)
    # print(get_dico())
    # names = list(get_dico().keys())
    # print(names)
    # print("in format")
    # print(list(reversed(names)))
