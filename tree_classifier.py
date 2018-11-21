dataset = [
    [['Зеленый', 3], 'Яблоко'],
    [['Желтый', 3], 'Яблоко'],
    [['Красный', 1], 'Виноград'],
    [['Красный', 1], 'Виноград'],
    [['Желтый', 3], 'Лимон'],
]

header = [['цвет', 'размер'], 'метка']


def unique_values(rows, col):
    return set([row[0][col] for row in rows])


def class_counts(dataset):
    counts = {}
    for row in dataset:
        label = row[1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def is_numeric(value):
    return isinstance(value, (int, float))


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "%s %s %s?" % (header[0][self.column], condition, str(self.value))


def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row[0]):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for label in counts:
        prob_label = counts[label] / float(len(rows))
        impurity -= prob_label ** 2
    return impurity


def info_gain(left, right, current):
    p = float(len(left)) / (len(left) + len(right))
    return current - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    best_gain = 0
    best_question = None
    current = gini(rows)
    n_features = len(rows[0][0])

    for col in range(n_features):
        values = set([row[0][col] for row in rows])

        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return Decision_node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Предположение", node.predictions)
        return
    print(spacing + str(node.question))
    print(spacing + "--> Да")
    print_tree(node.true_branch, spacing + '    ')
    print(spacing + "--> Нет")
    print_tree(node.false_branch, spacing + '    ')


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row[0]):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for label in counts.keys():
        probs[label] = str(int(counts[label] / total * 100)) + "%"
    return  probs





# print(find_best_split(dataset))
tree = build_tree(dataset)
# print_tree(tree)
# print(classify(dataset[0], tree))
# print(print_leaf(classify(dataset[1], tree)))

test = [
    [['Зеленый', 3]],
    [['Желтый', 4]],
    [['Зеленый', 2]],
    [['Зеленый', 1]],
    [['Желтый', 3]]
]

for row in test:
    print("Класс: %s" % (print_leaf(classify(row,tree))))
