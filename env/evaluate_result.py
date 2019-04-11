from keras.models import load_model, Model
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from keras import metrics
from itertools import cycle
from progressbar import ProgressBar
from processing.train_utils import imagenet_preprocess_input, wp_preprocess_input, id_preprocess_input, get_new_model
from processing.read_images import count_files
from os import listdir
from os.path import join
import numpy as np
from scipy import interp
from bokeh.plotting import figure, show, output_file
import pandas as pd
import os, json, pickle, csv, math, argparse

# resnet model
DEFAULT_MODEL_PATH = "models/resnet50_model/resnet50_06-0.517-2.090.hdf5"
VGG_MODEL_PATH = "models/vgg16_model/vgg16_01-0.520-1.567.hdf5"

DEFAULT_IMG_PATH = "data/van-gogh_sunflowers.jpg"
DEFAULT_TEST_PATH = "data/wikipaintings_small/wikipaintings_test"
DEFAULT_SAVE_PATH = "models/eval"
DATA_PATH = DEFAULT_IMG_PATH
N_CLASSES = 25

# (FROM RASTA) extended from evaluation.py to include additional statistics and evaluation tools

parser = argparse.ArgumentParser(description='Description')

parser.add_argument('-t', action="store", dest='type', help='Type of Evaluation [acc-predictive accuracy of model, pred-predict an image][acc|pred]')
parser.add_argument('-cv', action="store", dest='cv', help='Evaluate Cross Validation Output and Save [path to csv to save] to be used by train_hyp' )
parser.add_argument('-m', action="store", dest='model_path', default=DEFAULT_MODEL_PATH, help='Path of the model file')
parser.add_argument('--m_type', action="store", dest='model_type', choices=['resnet', 'vgg'], help="Choose the type of ready trained model to use for evaluation/prediction")
parser.add_argument('-d', action="store", dest="data_path", help="Path of test data")
parser.add_argument('-ds', action="store", dest="data_size", default= 's', choices=['f', 's'], help="Choose the size of test set, full or small")
parser.add_argument('-dp', action="store_true", dest='lab', help="Set to test in lab")
parser.add_argument('-k', action='store', dest='top_k', default='1,3,5', help='Top-k accuracy to compute')
parser.add_argument('-cm', action="store_true", dest='get_cm', default=False, help='Get Confusion Matrix')
parser.add_argument('--report', action="store_true", dest='get_class_report', default=False,
                    help='Get Classification Report')
parser.add_argument('--show', action="store_true", dest='show_g', default=False, help='Display graphs')
parser.add_argument('-s', action="store_true", dest='save', default=False, help='Save graphs')
parser.add_argument('--save', action="store", default=DEFAULT_SAVE_PATH, dest='save_path',
                    help='Specify save location')
parser.add_argument('--his', action="store", dest='plot_his', help='Plot history, choose which to plot [l|a|b (default)]')
parser.add_argument('-f', action="store", dest="file", help='Name of history file to plot: Reqruied for --his')
parser.add_argument('--model_name', action="store", dest='model_name', help='Model types/name: Required for --his')
parser.add_argument('--act', action="store", dest='act', help='Visualise activation function of layer [layer name or index]')
parser.add_argument('--roc', action="store_true", dest='get_roc', help='Get Roc Curve')
args = parser.parse_args()

# https://www.kaggle.com/amarjeet007/visualize-cnn-with-keras
# Generates activation map of filters in convolutional name 'layer_no' which can be given by
# index or by layer name
def get_act_map(model_path, img_path, target_size, layer_no, save, show, plot_size=(5,4), **kwargs):
    plot_size = (4,4)
    # target size is the size of the image to load in
    model = load_model(model_path)

    outputs = [layer.output for layer in model.layers]
    outputs = outputs[1:]
    act_model = Model(inputs=model.input, outputs=outputs)
    img = load_img(img_path, target_size=target_size)
    x = img_to_array(img)

    if 'vgg16' or 'resnet50' or 'inceptionv3' in model_path:
        x = imagenet_preprocess_input(x)
    else:
        x = wp_preprocess_input(x)

    new_x = x[np.newaxis,:,:,:]
    activations = act_model.predict(new_x, batch_size=1)
    index = None
    if layer_no.isdigit() or isinstance(layer_no, int):
        index = int(layer_no)
    else: #returned is name of layer
        for idx, layer in enumerate(model.layers):
            if layer.name == layer_no:
                index = idx
                break
    act = activations[index]
    i = 0

    f, ax = plt.subplots(plot_size[0], plot_size[1], squeeze=False)
    for r in range(0, plot_size[0]):
        for c in range(0, plot_size[1]):
            ax[r][c].imshow(act[0, :, :, i])
            i += 1
    if save:
        if 'path' in kwargs:
            path = kwargs['path']
        if 'name' in kwargs:
            name = kwargs['name']
        plt.savefig(path + '/' + name + '.svg', format='svg')
    if show:
        plt.show()

# (FROM RATSA) modified, given the test directory path, make predictions and collect
# top-k probabilities to be converted to string labels
# if model is weights only it uses default architecture for new model instance and load the
# weights
def get_y_prediction(model_path, test_path, top_k=1, target_size=(224, 224)):
    try:
        model = load_model(model_path)
    except ValueError: # when the given hdf5 saves weights only
        model, _ = get_new_model(args.model_name, (224,224,3), None, None, None, 0, 1, 0)
        model.load_weights(model_path)
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
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
            # use y_p to store original probability scores (on hold)
            y_pred.append(pred)

            y_pred_k.append([a for a in args_sorted[:top_k]])
            i += 1
            bar.update(i)
        y_true.append(y_t)
    return y_true, y_pred, np.asarray(y_pred_k)


# (FROM RASTA) calculates accuracy and outputs
def get_acc(model_path, test_path, k=[1, 3, 5], target_size=(224, 224), save=False, **kwargs):
    y_t, y_p ,y_pred = get_y_prediction(model_path, test_path, k[2], target_size)
    y_true = [j for sub in y_t for j in sub]
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
    if save:
        save_dict = {}
        if 'path' in kwargs:
            path = kwargs['path']
        if 'name' in kwargs:
            name = kwargs['name']
            save_dict['name'] = name
        save_dict['k'] = k
    for n, a in zip(k, acc):
        print("Top-{} accuracy: {}%".format(n, a * 100))
        if save:
            save_dict[str(n)] = str(a*100)
    if save:
        with open(join(path.rsplit('/', 1)[0], name + '-accuracy-output' + '.json'), 'w') as f:
            json.dump(save_dict, f)
    return y_t, y_p, one_true, one_pred, k, acc


# (FROM CS3099 ML7) computes confusion matrix
def get_confusion_matrix(y_true, y_pred, show, normalise=True, save=False, **kwargs):  # output of get_y_prediction
    dico = get_dico()
    cm = confusion_matrix(y_true, y_pred)
    orig_cm = cm
    if show:
        plt.figure(figsize=(18, 25))
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.title("Confusion matrix")
    classNames = [str(x) for x in list(dico.keys())]
    cm = pd.DataFrame(cm, columns=classNames, index=classNames)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues')
    #sns.heatmap(cm, cmap="YlGnBu", xticklabels=False, yticklabels=False)
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

# (FROM CS3099 ML7) Finds the optimal cut-off point which returns the optimal true positive rate with
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

# (FROM CS3099 ML7) Binarise the integer classes
def binarise(test, classes):
    test = label_binarize(test, classes=classes)
    return test

# (FROM CS3099 ML7) computes roc curve for multi class, for saving if prefer to separate
# plots for all classes add kwargs['all'] = True, else generate micro and macro average
# by default
def get_roc_curve(y_true, y_pred, save, path, show=False, **kwargs):
    k = 1
    nSamples = len(y_true)
    dico = get_dico()
    classes = [x for x in list(dico.values())]
    # Computes ROC curve and area for each class
    y_true = binarise(y_true, classes)
    y_pred = np.array(y_pred)
    y_pred = np.squeeze(y_pred)
    rocAuc = dict()
    fpr = dict()
    tpr = dict()

    for i in range(len(classes)):
        tprs = []
        meanFpr = np.linspace(0, 1, nSamples)
        # Computes by k fold cross validation where length of self.values_ denotes value of k
        jFpr, jTpr, _ = roc_curve(y_true[:,i].flatten(), y_pred[:,i].flatten())
        tprs.append(interp(meanFpr, jFpr, jTpr))
        tprs[-1][0] = 0.

        meanTpr = np.mean(tprs, axis=0)
        meanTpr[-1] = 1.0
        rocAuc[i] = dict()
        rocAuc[i]["area"] = round(auc(meanFpr, meanTpr), 2)
        fpr[i] = meanFpr
        tpr[i] = meanTpr
        optFpr, optTpr = find_opt_threshold(fpr[i], tpr[i])
        rocAuc[i]["optimal"] =  [("fpr", round(optFpr, 2)), ("tpr", round(optTpr, 2))]
    find_micro(y_true, y_pred, k, nSamples, fpr, tpr, rocAuc)
    find_macro(fpr, tpr, rocAuc, np.asarray(classes))

    name = None
    if save:
        name = kwargs['name']
    if show:
        if 'all' in kwargs:
            show_roc_curve_all(fpr, tpr, rocAuc, classes, name, path, show=True)
        else:
            show_roc_curve_bokeh(fpr, tpr, rocAuc, classes, name, path, show_plt=show)
    else:
        #print values of rocAuc instead!
        print(rocAuc)
    return rocAuc


# (FROM CS3099 ML7) computes micro average
def find_micro(y_true, y_pred, k, nSamples, fpr, tpr, rocAuc):
    # Computes micro-average ROC curve and area
    tprsMicro = []
    meanFpr = np.linspace(0, 1, nSamples)
    fprMicro, tprMicro, _ = roc_curve(y_true.ravel(), y_pred.ravel())
    tprsMicro.append(interp(meanFpr, fprMicro, tprMicro))
    tprsMicro[-1][0] = 0.
    meanTprMicro = np.mean(tprsMicro, axis=0)
    meanTprMicro[-1] = 1.0
    fpr["micro"] = meanFpr
    tpr["micro"] = meanTprMicro
    rocAuc["micro"] = round(auc(meanFpr, meanTprMicro), 2)


# (FROM CS3099 ML7) computes macro average
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


# use bokeh roc curve visualiser to to plot micro and macro average AUC
def show_roc_curve_bokeh(fpr, tpr, rocAuc, classes, name, path, show_plt=False):
    output_file("roc.html")
    from bokeh.models import Legend, LegendItem
    p = figure(plot_width=400, plot_height=400)

    # add a line renderer
    r = p.multi_line([fpr["micro"],fpr["macro"],[0,1]], [tpr["micro"], tpr["macro"],[0,1]],
                 color=["chocolate", "darkslateblue", "black"], alpha=[0.6, 0.6, 0.3], line_width=2)
    legend = Legend(items=[
        LegendItem(label='micro-average ROC curve (area = {0:0.3f})'
                       ''.format(rocAuc["micro"]), renderers=[r], index=0),
        LegendItem(label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(rocAuc["macro"]), renderers=[r], index=1),
    ])
    p.add_layout(legend)
    p.legend.location = "bottom_right"

    show(p)


# use matplotlib to plot individual roc curves for all classes
def show_roc_curve_all(fpr, tpr, rocAuc, classes, name, path, show=False):
    palette = cycle(sns.color_palette())
    classes = np.asarray(classes)
    for i in range(len(classes)):
        fig = plt.figure(figsize=[6, 4.5])
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.3f})'
                       ''.format(rocAuc["micro"]), color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(rocAuc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        plt.plot(fpr[i], tpr[i], color='aqua', lw=lw,
                  label='ROC curve of class {0} (area = {1:0.2f})'
                      ''.format(i+1, rocAuc[i]["area"]))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey',
                alpha=.2)
        plot_roc_grid('multi', name+"cl-"+str(i+1), path, fig, show=show)


# (FROM CS3099 ML7) plot the grid for matplotlib ROC curve plot
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
#
# # (FROM CS3099 ML7)
# def get_rates(y_true, y_pred, cm):
#     rates = dict()
#     row_sum = cm.sum(axis=1)
#     len = cm.shape[0]
#     total_sum = cm.sum()
#     ver_sum = cm.sum(axis=0)
#
#     for i in range(len):
#         output = []
#         tpr = cm[i][i] / row_sum[i]
#         fn = row_sum[i] - cm[i][i]
#         fp = ver_sum[i] - cm[i][i]
#         tnr = (total_sum - cm[i][i] - fn - fp) / (total_sum - row_sum[i])
#         # get classification accuracy and mis-classification
#         accuracy = accuracy_score(y_true, y_pred)
#         output.append(('accuracy', round(accuracy, 3)))
#         output.append(('error', round(1 - accuracy, 3)))
#         output.append(('sensitivity', round(tpr, 3)))
#         output.append(('specificity', round(tnr, 3)))
#         rates[i] = output
#     return rates

# (FROM RASTA) predict the style for a single image
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


# calculates classification report output including precision, recall, and f1 score
# saves as image with heatmap and a csv
def get_classification_report(y_true, y_pred, labels, name, title, show=True, save=False, **kwargs):
    ddl_heat = ['#DBDBDB', '#DCD5CC', '#DCCEBE', '#DDC8AF', '#DEC2A0', '#DEBB91',
                '#DFB583', '#DFAE74', '#E0A865', '#E1A256', '#E19B48', '#E29539']
    ddlheatmap = colors.ListedColormap(ddl_heat)
    title = title or 'Classification report'
    cr = classification_report(y_true, y_pred, target_names=labels)
    lines = cr.split('\n')
    classes = []
    matrix = []
    for line in lines[2:(len(lines) - 5)]:
        s = line.split()
        classes.append(s[0])
        value = [float(x) for x in s[1: len(s) - 1]]
        matrix.append(value)

    plt.figure(figsize=(18, 23))
    matrix = pd.DataFrame(matrix, columns=['precision', 'recall', 'f1-score'], index=classes)
    ax = sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues')

    ax.figure.tight_layout()
    plt.title(title)
    plt.ylabel('Classes')
    plt.xlabel('Measures')

    if save:
        if 'path' in kwargs:
            path = kwargs['path']
        plt.savefig(path + '/class_report_' + name + '.svg', format='svg', bbox_inches='tight')
        matrix.to_csv(join(path, name + 'classification_report.csv'))
    if show:

        plt.show()

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

# display scaled image
def show_image(img_path):
    img = load_img(img_path, target_size=(224,224))
    x = img_to_array(img)
    plt.imshow(x/255.)
    plt.title('Original image')
    plt.savefig(img_path.rsplit('/', 1)[1] + '_scaled.jpg')
    plt.show()

# image try to reconstruct it using the autoencoder
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
# (FROM RASTA) returns class indices for corresponding class label
def get_dico():
    classes = []
    PATH = os.path.dirname(__file__)
    #directory = join(PATH, 'data/wikipaintings_small/wikipaintings_train')
    directory = "/cs/tmp/yk30/data/wikipaintings_full/wikipaintings_train"
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)
    class_indices = dict(zip(classes, range(len(classes))))
    return class_indices

# invert the dictionary to be value:label
def invert_dico(dico):
    return {v: k for k, v in dico.items()}

# write the output to csv
def write_to_csv(path, data, type):
    FIELDNAMES = list(data.keys()).sort()
    ordered_dict = OrderedDict(sorted(data.items(), key=lambda t: t[0]))
    with open(join(path, '_'+type+'.csv'), 'a+', newline='') as f:
        lines = f.readlines()
        l = len(lines)
        head = True
        if l > 1:
            head = False
        w = csv.DictWriter(f, ordered_dict.keys())
        if head:
            w.writeheader()
        w.writerow(ordered_dict)


# main method decoding user command line options and use respective method calls
# in the case of -t acc or pred only
def evaluate():
    MODEL_PATH = args.model_path
    if args.model_type is not None:
        if args.model_type=='vgg':
            MODEL_PATH = VGG_MODEL_PATH
    PRE_PATH = ''
    k = (str(args.top_k)).split(",")
    K = [int(val) for val in k]
    SAVE = args.save
    if SAVE:
        SAVE_PATH = args.save_path
    else:
        SAVE_PATH = None
    if args.type == 'acc':
        if args.lab:
            PRE_PATH = '/cs/tmp/yk30/'
        if args.data_path is not None:
            DATA_PATH = args.data_path
        else:
            if args.data_size is not None and args.data_size=='f':
                DATA_PATH = PRE_PATH + "data/wikipaintings_full/wikipaintings_test"
            elif args.data_size == 's':
                DATA_PATH = DEFAULT_TEST_PATH

        name = MODEL_PATH.rsplit('/', 1)[1].replace('.hdf5', '')
        if args.model_name is not None:
            name = name + '-' + args.model_name
        SHOW = args.show_g
        y_t, y_p, y_true, y_pred, k, acc = get_acc(MODEL_PATH, DATA_PATH, k=K, save=SAVE, path=SAVE_PATH, name=name)

        if args.get_cm:
            cm = get_confusion_matrix(y_true, y_pred, show=SHOW, save=SAVE, path=SAVE_PATH, name=name)
        if args.get_class_report:
            classes = get_dico().keys()
            get_classification_report(y_true, y_pred, classes, name, 'classification report: ' + name, show=SHOW,
                                      save=SAVE,
                                      path=SAVE_PATH)
        if args.get_roc:
            get_roc_curve(y_true, y_p, show=SHOW, save=SAVE, path=SAVE_PATH, name=name)
        if args.cv:
            csv_path = args.cv
            dic = {(key, value) for (key, value) in zip(k, acc)}
            write_to_csv(csv_path, dic,'cv')
    else:
        DATA_PATH = args.data_path
        if args.data_path is None:
            DATA_PATH = DEFAULT_IMG_PATH
        v = 5
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

# method for any options to plot history or generating activation maps
def plot():
    if args.model_name is None:
        if args.model_type is not None:
            args.model_name = args.model_type
        else: args.model_name = ""

    if args.plot_his is not None:
        his_t = args.plot_his
        his = pickle.load(open(args.file, 'rb'))
        if his_t == 'b':
            his_t = 'a'
        plot_history_plotly(his, his_type[his_t], save=args.save, path=args.save_path, name=args.model_name+"-"+his_type[his_t]+" Plot", show=args.show_g)
        if args.plot_his == 'b':
            his_t = 'l'
            plot_history_plotly(his, his_type[his_t], save=args.save, path=args.save_path, name=args.model_name + "-" + his_type[his_t] + " Plot", show=args.show_g)
    elif args.act is not None:
        if args.model_type is not None:
            if args.model_type=='vgg':
                MODEL_PATH = VGG_MODEL_PATH
        else:
            MODEL_PATH = args.model_path
        DATA_PATH = args.data_path
        if DATA_PATH is None:
            DATA_PATH = DEFAULT_IMG_PATH
        layer_no = args.act
        get_act_map(MODEL_PATH, DATA_PATH, (224, 224), layer_no=layer_no, save=args.save, path=args.save_path, name=args.model_name + "-" + layer_no + "-act-map", show=args.show_g)

if args.type is not None:
    evaluate()
else:
    plot()
