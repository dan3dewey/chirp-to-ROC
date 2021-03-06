import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf as scipy_erf


def make_chirp(n_out=400, nhalfcycles=6.5, warpexp=0.65,
               symmetric=False, noise=0):
    """Create two regions with a "diagonal chirp" boundary
      Parameters:
          n_out -- number of generated samples returned
    nhalfcycles -- half-cycles in one side of the chirp
        warpexp -- exponent used to warp the chirp
      symmetric -- determines a central chirp (True) or central node (False)
          noise -- amplitude of gaussian noise to blur x values.
      Returns a tuple: x_out, y_out
    """

    # Symmetry determines if we use sin or cos
    if symmetric:
        trigfunc = np.cos
    else:
        trigfunc = np.sin

    # we'll lose about 1/2 of the points, so include some extra
    n_samples = 2 * int(n_out + 5 * np.sqrt(n_out))
    # x1, x2 are uniform -1 to 1:
    x1 = 2.0 * np.random.rand(n_samples) - 1.0
    x2 = 2.0 * np.random.rand(n_samples) - 1.0
    # plt.scatter(x1,x2)

    # warp the x1 scale, preserving -1,1 --> -1,1
    x1warp = (abs(x1))**warpexp * x1 / abs(x1)
    # determine the boundary between the two classes
    x2boundary = trigfunc(x1warp * nhalfcycles * np.pi) * (1.0 - abs(x1warp))
    # plt.scatter(x1,x2boundary)

    y_class = x2 > x2boundary
    ##plt.scatter(x1, x2, c=y_class, s=40, cmap=plt.cm.Spectral);

    # rotate x1, x2 by 45 deg (also scales by sqrt(2))
    # and add noise (blurring) if desired
    x1rot = x1 - x2
    x2rot = x1 + x2
    if noise > 0:
        x1rot += noise * np.random.randn(len(x1rot))
        x2rot += noise * np.random.randn(len(x2rot))
    ##plt.scatter(x1rot, x2rot, c=y_class, s=40, cmap=plt.cm.Spectral);

    # and keep just the central square
    x1out = []
    x2out = []
    yout = []
    for isamp in range(len(x1rot)):
        if (abs(x1rot[isamp]) <= 1.0) and (abs(x2rot[isamp]) <= 1.0):
            x1out.append(x1rot[isamp])
            x2out.append(x2rot[isamp])
            yout.append(y_class[isamp])
    percent1 = int(10000 * sum(yout) / len(yout)) / 100
    # print("Number of output samples =",len(yout),
    # " Percent split: ", percent1, "-", 100-percent1)
    ##plt.scatter(x1out, x2out, c=yout, s=40, cmap=plt.cm.Spectral);

    # Return X and Y in scikit-learn standard shapes
    out_X = np.zeros((n_out, 2))
    out_X[:, 0] = x1out[0:n_out]
    out_X[:, 1] = x2out[0:n_out]
    out_Y = np.zeros((n_out, 1))
    out_Y[:, 0] = yout[0:n_out]
    return out_X, out_Y.squeeze()


def chirp_region(x_in, nhalfcycles=6.5, warpexp=0.65,
                 symmetric=False, noise=None):
    """Determine which region of the chirp the x_in vectors are in
    by assigning a probability of 0 to 1.
    If the noise is non-zero the erf() is used to approximate
    the probability, otherwise it is binary 0 or 1.
    This is very close to the "best" classifier possible for the noisy chirp.
    Parameters:
        x_in -- usual set of ML X vectors with shape of (m, 2)
        others as in make_chirp().
    """
    # confirm some of the x_in dimension properties
    assert((len(x_in.shape) == 2) and (x_in.shape[1] == 2))
    # convert the input xs (similar to the x1rot,x2rot above)
    # to the diagonal coordinate system: the x1, x2 above.
    x1r = x_in[:, 0]
    x2r = x_in[:, 1]
    x1 = (x1r + x2r) / 2.0
    x2 = (x2r - x1r) / 2.0
    y_out = np.zeros(len(x1))
    # go from x1 to the x2boundary
    x1warp = (abs(x1))**warpexp * x1 / abs(x1)
    # Symmetry determines if we use sin or cos
    if symmetric:
        trigfunc = np.cos
    else:
        trigfunc = np.sin

    # determine the boundary (in x2) between the two classes (at x1)
    x2boundary = trigfunc(x1warp * nhalfcycles * np.pi) * (1.0 - abs(x1warp))
    if noise > 0:
        # Use erf() to assign a 0 to 1 probability
        zboundary = (x2 - x2boundary)/noise
        y_class = 0.5 * scipy_erf(zboundary / np.sqrt(2)) + 0.5
    else:
        # assign binary values
        y_class = 1 * (x2 > x2boundary)
    # show the assignment in the input coord.s:
    ##plt.scatter(x1r, x2r, s=10, c=y_class, cmap=plt.cm.Spectral);
    return y_class


def plot_Xy(X_in, y_in, title="2D Xs with y color coding",
            s=25):
    """Plot the Xs in x0,x1 space and color-code by the ys.
    """
    # set figure limits
    xmin = -1.0
    xmax = 1.0
    ymin = -1.0
    ymax = 1.0

    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([ymin, ymax])
    plt.scatter(X_in[:, 0], X_in[:, 1],
                c=y_in.squeeze(), s=s, cmap=plt.cm.Spectral)


def y_yhat_plots(y, yh, title="y and y_score", y_thresh=0.5):
    """Output plots showing how y and y_hat are related:
    the "confusion dots" plot is analogous to the confusion table,
    and the standard ROC plot with its AOC value.
    The y=1 threshold can be changed with the y_thresh parameter.
    """
    # The predicted y value with threshold = y_thresh
    y_pred = 1.0 * (yh > y_thresh)

    # Show table of actual and predicted counts
    crosstab = pd.crosstab(y, y_pred, rownames=[
                           'Actual'], colnames=['  Predicted'])
    print("\nConfusion matrix (y_thresh={:.3f}):\n\n".format(y_thresh),
        crosstab)

    # Calculate the various metrics and rates
    tn = crosstab[0][0]
    fp = crosstab[1][0]
    fn = crosstab[0][1]
    tp = crosstab[1][1]

    ##print(" tn =",tn)
    ##print(" fp =",fp)
    ##print(" fn =",fn)
    ##print(" tp =",tp)

    this_fpr = fp / (fp + tn)
    this_fnr = fn / (fn + tp)

    this_recall = tp / (tp + fn)
    this_precision = tp / (tp + fp)
    this_accur = (tp + tn) / (tp + fn + fp + tn)

    this_posfrac = (tp + fn) / (tp + fn + fp + tn)

    print("\nResults:\n")
    print(" False Pos = ", 100.0 * this_fpr, "%")
    print(" False Neg = ", 100.0 * this_fnr, "%")
    print("    Recall = ", 100.0 * this_recall, "%")
    print(" Precision = ", 100.0 * this_precision, "%")
    print("\n    Accuracy = ", 100.0 * this_accur, "%")
    print(" Pos. fract. = ", 100.0 * this_posfrac, "%")

    # Put them in a dataframe
    ysframe = pd.DataFrame([y, yh, y_pred], index=[
                           'y', 'y-hat', 'y-pred']).transpose()

    # If the yh is discrete (0 and 1s only) then blur it a bit
    # for a better visual dots plot
    if min(abs(yh - 0.5)) > 0.49:
        ysframe["y-hat"] = (0.51 * ysframe["y-hat"]
                            + 0.49 * np.random.rand(len(yh)))

    # Make a "confusion dots" plot
    # Add a blurred y column
    ysframe['y (blurred)'] = y + 0.1 * np.random.randn(len(y))

    # Plot the real y (blurred) vs the predicted probability
    # Note the flipped ylim values.
    ysframe.plot.scatter('y-hat', 'y (blurred)', figsize=(12, 5),
                         s=2, xlim=(0.0, 1.0), ylim=(1.8, -0.8))
    # show the "correct" locations on the plot
    plt.plot([0.0, y_thresh], [0.0, 0.0], '-',
        color='green', linewidth=5)
    plt.plot([y_thresh, y_thresh], [0.0, 1.0], '-',
        color='gray', linewidth=2)
    plt.plot([y_thresh, 1.0], [1.0, 1.0], '-',
        color='green', linewidth=5)
    plt.title("Confusion-dots Plot: " + title, fontsize=16)
    # some labels
    ythr2 = y_thresh/2.0
    plt.text(ythr2 - 0.03, 1.52, "FN", fontsize=16, color='red')
    plt.text(ythr2 + 0.5 - 0.03, 1.52, "TP", fontsize=16, color='green')
    plt.text(ythr2 - 0.03, -0.50, "TN", fontsize=16, color='green')
    plt.text(ythr2 + 0.5 - 0.03, -0.50, "FP", fontsize=16, color='red')

    plt.show()

   # Make the ROC curve
    # Set the y-hat as the index and sort on it
    ysframe = ysframe.set_index('y-hat').sort_index()
    # Put y-hat back as a column (but the sorting remains)
    ysframe = ysframe.reset_index()

    # Initialize the counts for threshold = 0
    p_thresh = 0
    FN = 0
    TN = 0
    TP = sum(ysframe['y'])
    FP = len(ysframe) - TP

    # Assemble the fpr and recall values
    recall = []
    fpr = []
    # Go through each sample in y-hat order,
    # advancing the threshold and adjusting the counts
    for iprob in range(len(ysframe['y-hat'])):
        p_thresh = ysframe.iloc[iprob]['y-hat']
        if ysframe.iloc[iprob]['y'] == 0:
            FP -= 1
            TN += 1
        else:
            TP -= 1
            FN += 1
        # Recall and FPR:
        recall.append(TP / (TP + FN))
        fpr.append(FP / (FP + TN))

    # Put recall and fpr in the dataframe
    ysframe['Recall'] = recall
    ysframe['FPR'] = fpr

    # - - - ROC - - - could be separate routine
    zoom_in = False

    # Calculate the area under the ROC
    roc_area = 0.0
    for ifpr in range(1, len(fpr)):
        # add on the bit of area (note sign change, going from high fpr to low)
        roc_area += 0.5 * (recall[ifpr] + recall[ifpr - 1]
                           ) * (fpr[ifpr - 1] - fpr[ifpr])

    plt.figure(figsize=(8, 8))
    plt.title("ROC: " + title, size=16)
    plt.plot(fpr, recall, '-b')
    # Set the scales
    if zoom_in:
        plt.xlim(0.0, 0.10)
        plt.ylim(0.0, 0.50)
    else:
        # full range:
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)

    # The reference line
    plt.plot([0., 1.], [0., 1.], '--', color='orange')

    # The point at the y_hat = y_tresh threshold
    if True:
        plt.plot([this_fpr], [this_recall], 'o', c='blue', markersize=15)
        plt.xlabel('False Postive Rate', size=16)
        plt.ylabel('Recall', size=16)
        plt.annotate('y_hat = {:.2f}'.format(y_thresh),
                            xy=(this_fpr + 0.015,
                            this_recall), size=14, color='blue')
        plt.annotate(' Pos.Fraction = ' +
                        '  {:.0f}%'.format(100 * this_posfrac),
                        xy=(this_fpr + 0.02, this_recall - 0.03),
                        size=14, color='blue')

    # Show the ROC area (shows on zoomed-out plot)
    plt.annotate('ROC Area = ' + str(roc_area)
                 [:5], xy=(0.4, 0.1), size=16, color='blue')

    # Show the plot
    plt.show()

    return ysframe
