import pandas as pd
from matplotlib import pyplot
from sklearn import linear_model, metrics, model_selection, preprocessing, svm

url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/mushrooms.csv"
df = pd.read_csv(url)

le = preprocessing.LabelEncoder()

df['type'] = le.fit_transform(df['type'])
df['cap_shape'] = le.fit_transform(df['cap_shape'])
df['cap_surface'] = le.fit_transform(df['cap_surface'])
df['cap_color'] = le.fit_transform(df['cap_color'])
df['bruises'] = le.fit_transform(df['bruises'])
df['odor'] = le.fit_transform(df['odor'])
df['gill_attachment'] = le.fit_transform(df['gill_attachment'])
df['gill_spacing'] = le.fit_transform(df['gill_spacing'])
df['gill_size'] = le.fit_transform(df['gill_size'])
df['gill_color'] = le.fit_transform(df['gill_color'])
df['stalk_shape'] = le.fit_transform(df['stalk_shape'])
df['stalk_root'] = le.fit_transform(df['stalk_root'])
df['stalk_surface_above_ring'] = le.fit_transform(
    df['stalk_surface_above_ring'])
df['stalk_surface_below_ring'] = le.fit_transform(
    df['stalk_surface_below_ring'])
df['stalk_color_above_ring'] = le.fit_transform(df['stalk_color_above_ring'])
df['stalk_color_below_ring'] = le.fit_transform(df['stalk_color_below_ring'])
df['veil_type'] = le.fit_transform(df['veil_type'])
df['veil_color'] = le.fit_transform(df['veil_color'])
df['ring_number'] = le.fit_transform(df['ring_number'])
df['ring_type'] = le.fit_transform(df['ring_type'])
df['spore_print_color'] = le.fit_transform(df['spore_print_color'])
df['population'] = le.fit_transform(df['population'])
df['habitat'] = le.fit_transform(df['habitat'])

svm_model = svm.SVC(max_iter=2000)
log_model = linear_model.LogisticRegression(max_iter=2000)

y = df.values[:, 0]
X = df.values[:, 1:]

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.25)

svm_model.fit(X_train, y_train)
log_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
y_pred_log = log_model.predict(X_test)


print(
    f"\n\nSupport Vector Machine Accuracy Score: {metrics.accuracy_score(y_test, y_pred_svm)}")
print(
    f"\nLogistic Regression Accuracy Score: {metrics.accuracy_score(y_test, y_pred_log)}\n")


######################### V I S U A L I Z A T I O N S #########################
# HISTOGRAM
# df.hist()

# CONFUSION MATRIX
# metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm)

# CORRELATION MATRIX
# fig = pyplot.figure(figsize=(20, 20))
# pyplot.matshow(df.corr(), fignum=fig.number)
# pyplot.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(
#     ['number']).columns, fontsize=7, rotation=80)
# pyplot.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(
#     ['number']).columns, fontsize=7)
# cb = pyplot.colorbar()
# cb.ax.tick_params(labelsize=7)

# pyplot.show()


print("\n/ / / / / / / / / / / / / / / / / / / / / / / / / / /")
print("/                                                   /")
print("/        WHAT KIND OF MUSHROOM DO YOU HAVE?         /")
print("/                                                   /")
print("/ / / / / / / / / / / / / / / / / / / / / / / / / / /\n")

input("\nPress Enter to continue...\n")

categoryList = ["bell = b, \nconical = c, \nconvex = x, \nflat = f, \nknobbed = k, \nsunken = s\n\n(1/22) CAP-SHAPE: ", "fibrous = f, \ngrooves = g, \nscaly = y, \nsmooth = s\n\n(2/22) CAP-SURFACE: ", "brown = n, \nbuff = b, \ncinnamon = c, \ngray = g, \ngreen = r, \npink = p, \npurple = u, \nred = e, \nwhite = w, \nyellow = y\n\n(3/22) CAP-COLOR: ", "bruises = t, \nno = f\n\n(4/22) BRUISES: ", "almond = a, \nanise = l, \ncreosote = c, \nfishy = y, \nfoul = f, \nmusty = m, \nnone = n, \npungent = p, \nspicy = s\n\n(5/22) ODOR: ", "attached = a, \ndescending = d, \nfree = f, \nnotched = n\n\n(6/22) GILL-ATTACHMENT: ", "close = c, \ncrowded = w, \ndistant = d\n\n(7/22) GILL-SPACING: ", "broad = b, \nnarrow = n\n\n(8/22) GILL-SIZE: ", "black = k, \nbrown = n, \nbuff = b, \nchocolate = h, \ngray = g, \ngreen = r, \norange = o, \npink = p, \npurple = u, \nred = e, \nwhite = w, \nyellow = y\n\n(9/22) GILL-COLOR: ", "enlarging = e, \ntapering = t\n\n(10/22) STALK-SHAPE: ", "bulbous = b, \nclub = c, \ncup = u, \nequal = e, \nrhizomorphs = z, \nrooted = r, \nmissing = ?\n\n(11/22) STALK-ROOT: ", "fibrous = f, \nscaly = y, \nsilky = k, \nsmooth = s\n\n(12/22) STALK-SURFACE-ABOVE-RING: ",
                "fibrous = f, \nscaly = y, \nsilky = k, \nsmooth = s\n\n(13/22) STALK-SURFACE-BELOW-RING: ", "brown = n, \nbuff = b, \ncinnamon = c, \ngray = g, \norange = o, \npink = p, \nred = e, \nwhite = w, \nyellow = y\n\n(14/22) STALK-COLOR-ABOVE-RING: ", "brown = n, \nbuff = b, \ncinnamon = c, \ngray = g, \norange = o, \npink = p, \nred = e, \nwhite = w, \nyellow = y\n\n(15/22) STALK-COLOR-BELOW-RING: ", "partial = p, \nuniversal = u\n\n(16/22) VEIL-TYPE: ", "brown = n, \norange = o, \nwhite = w, \nyellow = y\n\n(17/22) VEIL-COLOR: ", "none = n, \none = o, \ntwo = t\n\n(18/22) RING-NUMBER: ", "cobwebby = c, \nevanescent = e, \nflaring = f, \nlarge = l, \nnone = n, \npendant = p, \nsheathing = s, \nzone = z\n\n(19/22) RING-TYPE: ", "black = k, \nbrown = n, \nbuff = b, \nchocolate = h, \ngreen = r, \norange = o, \npurple = u, \nwhite = w, \nyellow = y\n\n(20/22) SPORE-PRINT-COLOR: ", "abundant = a, \nclustered = c, \nnumerous = n, \nscattered = s, \nseveral = v, \nsolitary = y\n\n(21/22) POPULATION: ", "grasses = g, \nleaves = l, \nmeadows = m, \npaths = p, \nurban = u, \nwaste = w, \nwoods = d\n\n(22/22) HABITAT: "]

valueList = []


for i in range(22):
    print("\n-----------------------------------------------------\n")

    options = categoryList[i].split("= ")
    mySet = set()

    for j in range(1, len(options)):
        mySet.add(options[j][0])

    # All possible choices are now inside mySet

    value = input(categoryList[i])

    while (value not in mySet):
        print("Please make a valid selection.\n")
        value = input(categoryList[i].split("\n\n")[1])

    valueList.append(value)

valueList = le.fit_transform(valueList)

# 0 = edible
# 1 = poisonous

res = svm_model.predict([valueList])


def category(res): return "edible" if res == 0 else "poisonous"


print("\n---------------------------------------------------\n\n")
print(f"Your mushroom is {category(res)}... probably\n\n")
