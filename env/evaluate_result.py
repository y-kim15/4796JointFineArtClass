from keras.models import load_model, Model
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
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
import pickle
import itertools


MODEL_PATH = "models/resnet50_1-24-13-58_empty_tune-3-no-0/retrain-tune-3/19-0.343._retrain_layers-3-s-1/12-0.408.hdf5"
# IMG_PATH = "../data/wikipaintings_full/wikipaintings_test/Baroque/adriaen-brouwer_village-barbershop.jpg"
# get_act_map(MODEL_PATH, IMG_PATH, target_size=(224, 224), layer_no=100, plot_size=(8, 8))

DEFAULT_MODEL_PATH = MODEL_PATH
DEFAULT_SAVE_PATH = "models/eval"
N_CLASSES = 25

parser = argparse.ArgumentParser(description='Description')

# parser.add_argument('-t', action="store", default='acc', dest='type', help='Type of Evaluation [acc-predictive accuracy of model]')
parser.add_argument('-m', action="store", dest='model_path', default=DEFAULT_MODEL_PATH, help='Path of the model file')
parser.add_argument('-d', action="store", dest="data_path",
                    default="data/wikipaintings_small/wikipaintings_test", help="Path of test data")
parser.add_argument('-k', action='store', dest='top_k', default='1,3,5', help='Top-k accuracy to compute')
parser.add_argument('-cm', action="store_true", dest='get_cm', default=False, help='Get Confusion Matrix')
parser.add_argument('-pr', action="store_true", dest='get_pr', default=False, help='Get Precision Recall Curve')
parser.add_argument('--report', action="store_true", dest='get_class_report', default=False,
                    help='Get Classification Report')
parser.add_argument('-show', action="store_true", dest='show_g', default=False, help='Display graphs')
parser.add_argument('-s', action="store_true", dest='save', default=False, help='Save graphs')
parser.add_argument('--save', action="store", default=DEFAULT_SAVE_PATH, dest='save_path',
                    help='Specify save location')
# parser.add_argument('--dropout', action="store", default=0.0, type=float, dest='add_drop', help='Add dropout rate [0-1]')
# parser.add_argument('--mom', action="store", default=0.0, type=float, dest='add_mom', help='Add momentum to SGD')
# parser.add_argument('-ln', action="store", type=int, dest='layer_no', help='Select the layer to replace')
args = parser.parse_args()


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
    activations = act_model.predict(x[np.newaxis, :, :, :])
    act = activations[layer_no]
    i = 0
    fig, ax = plt.subplot(plot_size[0], plot_size[1], figsize=(plot_size[0] * 2.5, plot_size[1] * 1.5))
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


def get_acc(model_path, test_path, k=[1, 3, 5], target_size=(224, 224)):
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
        for preds, v in zip(y_pred[:, :n], y_true):
            if v in preds:
                out += 1
        acc.append(out / max)
    for n, a in zip(k, acc):
        print("Top-{} accuracy: {}%".format(n, a * 100))
    return one_pred, one_true


def get_confusion_matrix(y_true, y_pred, show, normalise=True, save=False, **kwargs):  # output of get_y_prediction
    dico = get_dico()
    cm = confusion_matrix(y_true, y_pred)
    orig_cm = cm
    if show:
        plt.figure(figsize=(16, 14))
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.1f'
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

    return orig_cm


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
    preds = [inv_dico.get(a) for a in args_sorted[:top_k]]
    pcts = [pred[0][a] for a in args_sorted[:top_k]]
    return preds, pcts


def get_precision_recall(y_true, y_pred, name, display_all=False):
    values = []
    classes = list(range(N_CLASSES))
    print("classes ", classes)
    '''
    for i in range(len(y_true)):
        print("i: ", i)
        print(y_true)
        enc = label_binarize(y_true[i], classes=classes)
        print(enc)
        t_bin.append(enc)
    '''
    t_bin = to_categorical(y_true)
    p_bin = to_categorical(y_pred)
    print(t_bin[0], y_pred[0])
    for (t, p) in zip(t_bin, p_bin):
        precision, recall, _ = precision_recall_curve(t.ravel(), p.ravel())
        values.append((precision, recall))
    all_true = np.concatenate(t_bin)
    all_pred = np.concatenate(p_bin)
    all_precision, all_recall, _ = precision_recall_curve(all_true.ravel(), all_pred.ravel())
    area = round(auc(all_recall, all_precision), 2)
    if not display_all:
        x = np.linspace(0.1, 5, 80)

        p = figure(title='Precision & Recall for ' + name, y_axis_type=float,
                   x_range=(0.0, max(all_recall) + 1.0), y_range=(0.0, max(all_precision) + 1.0),
                   background_fill_color="#fafafa")

        p.line(all_recall, all_precision, legend='Overall AUC %.4f' % (round(auc(all_recall, all_precision, 2))),
               line_color="tomato", line_dash="dashed")
        p.legend.location = "top_left"
        file_name = 'pr_curve_' + name
        output_file(file_name + ".html", title=file_name)
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


def plot_his(history):
    loss_list = history['loss']
    val_loss_list = history['val_loss']
    acc_list = history['acc']
    val_acc_list = history['val_acc']
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return
        ## As loss always exists
    epochs = list(range(1, len(loss_list) + 1))
    plt.subplots(figsize=(17, 10))
    sns.set_style('darkgrid')
    data = {'Epochs': epochs*2, 'Type': list(itertools.chain.from_iterable(itertools.repeat(x, len(epochs)) for x in ['train', 'val'])),
            'Loss': loss_list + val_loss_list, 'Metric':['Loss']*len(epochs)*2}
    df = pd.DataFrame(data)
    sns.lineplot(x='Epochs', y='Loss', hue='Type', data=df)
    plt.show()

    plt.subplots(figsize=(17,10))
    sns.set()
    data2 = {'Epochs': epochs * 2,
            'Type': list(itertools.chain.from_iterable(itertools.repeat(x, len(epochs)) for x in ['train', 'val'])),
            'Accuracy':  acc_list + val_acc_list, 'Metric': list(
            itertools.chain.from_iterable(itertools.repeat(x, len(epochs*2)) for x in ['Accuracy']))}
    df2 = pd.DataFrame(data2)
    sns.lineplot(x='Epochs', y='Accuracy', hue='Type', data=df2)
    #g = sns.FacetGrid(df, col='Metric', hue='Type', sharey=False)
    #g.map(sns.lineplot, 'Epochs', 'Value')

    plt.show()


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
    plt.plot(epochs, acc_list, 'b', label='Training accuracy (' + str(format(acc_list[-1], '.5f')) + ')')
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
    directory = join(PATH, 'data/wikipaintings_small/wikipaintings_train')
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)
    class_indices = dict(zip(classes, range(len(classes))))
    return class_indices


def invert_dico(dico):
    return {v: k for k, v in dico.items()}

def evaluate():
    MODEL_PATH = args.model_path
    DATA_PATH = args.data_path
    k = (str(args.top_k)).split(",")
    K = [int(val) for val in k]
    y_pred, y_true = get_acc(MODEL_PATH, DATA_PATH, k=K)
    name = MODEL_PATH.rsplit('/', 1)[1].replace('.hdf5', '')
    SHOW = args.show_g
    SAVE = args.save
    if SAVE:
        SAVE_PATH = args.save_path
        #SAVE_PATH = join(args.save_path, name)
        #create_dir(join(SAVE_PATH, name))
    else:
        SAVE_PATH = None

    if args.get_cm:
        cm = get_confusion_matrix(y_true, y_pred, show=SHOW, save=SAVE, path=SAVE_PATH, name=name)
    if args.get_class_report:
        classes = get_dico().keys()
        get_classification_report(y_true, y_pred, classes, name, 'classification report: ' + name, show=SHOW, save=SAVE,
                                  path=SAVE_PATH)
    if args.get_pr:
        get_precision_recall(y_true, y_pred, name, SHOW)
evaluate()
#his = pickle.load(open('models/resnet50_1-24-13-58_empty_tune-3-no-0/retrain-tune-3/19-0.343._retrain_layers-3-s-1/'
#                       '12-0.408._retrain_layers-3-s-4/history.pck', 'rb'))
#print(his)
#plot_history_hover(his, 'Accuracy', save=True, path='models/eval/resnet50_1-24-13-58_empty_tune-3-no-0', name='resnet50_retrained_12-0.40_acc')
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
