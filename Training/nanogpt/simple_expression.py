import random
import copy
from collections import Counter

# tree class with additional methods
class TreeNode:
    def __init__(self, value, fixed = None):
        self.value = value
        self.left = None
        self.right = None
        if fixed :
            self.set_fixed(fixed)

    # set as given string
    def set_fixed(self, fixed):
        self.value = fixed[0]
        if self.value in ['+','*','^'] :
            remaining_leaves = 1
            for i in range(len(fixed)-1):
                if fixed[i+1] in ['+','*','^'] :
                    remaining_leaves += 1
                else :
                    remaining_leaves -= 1
                if remaining_leaves == 0 :
                    right_position = i+2
                    break
            self.left = TreeNode(fixed[1],fixed[1:right_position])
            self.right = TreeNode(fixed[right_position],fixed[right_position:])
            return
        else :
            return

    # string generated with prefix notation
    def __str__(self):
        return self.stringify_prefix()

    # prefix notation
    def stringify_prefix(self):
        if self.left == None and self.right == None:
            return str(self.value)

        result = self.value
        for child in [self.left, self.right]:
            result += " " + child.stringify_prefix()
        return result

    # randomly generates expression represented in a tree
    @staticmethod
    def generate_expression(max_depth, TOTAL_VARIABLES):
        if max_depth == 1:
            return TreeNode(random.choice(list(TOTAL_VARIABLES)))
        else:
            operator = random.choice(['+', '*'])
            node = TreeNode(operator)

            if operator == '^':
                node.left = TreeNode.generate_expression(max_depth - 1, TOTAL_VARIABLES)
                node.right = TreeNode(random.randint(1, 5))
            else:
                left_depth = random.randint(1, max_depth - 1)
                right_depth = random.randint(1, max_depth - 1)
                node.left = TreeNode.generate_expression(left_depth, TOTAL_VARIABLES)
                node.right = TreeNode.generate_expression(right_depth, TOTAL_VARIABLES)
            return node

    # substitutes all variables within a dictionary
    def substitute_variable(self, sub):
        if self.value in sub:
            return sub[self.value]

        if self.left:
            self.left = self.left.substitute_variable(sub)

        if self.right:
            self.right = self.right.substitute_variable(sub)

        return self

    # infix notation
    def stringify_infix(self):
        if self.left and self.right:
            return f"({self.left.stringify_infix()} {self.value} {self.right.stringify_infix()})"
        else:
            return str(self.value)

    # expand expression e.g. (a+b)*(c+d) -> a*c + a*d + b*c + b*d
    def expansion(self):
        # depth
        if self.left:
            self.left = self.left.expansion()
        if self.right:
            self.right = self.right.expansion()

        if self.value == '*' :
            left = []
            right = []
            if self.left.value == '+' :
                left.append(self.left.left)
                left.append(self.left.right)
            else :
                left.append(self.left)

            if self.right.value == '+' :
                right.append(self.right.left)
                right.append(self.right.right)
            else :
                right.append(self.right)

            num_terms = len(left) * len(right)
            result = '+'*(num_terms-1)

            for l in left :
                for r in right :
                    result = result + '*' + str(l) + str(r)
                    result = result.replace(" ", "")

            return TreeNode(result[0],result)

        else :
            return self

    # constructs single tree from monomials and their coefficients
    def combine_monomials_coefficients(self, collection):
        if not collection:
            return None

        def combine_trees(operator, tree_1, tree_2):
            res = TreeNode(operator)
            res.left = tree_1
            res.right = tree_2

            return res

        monomial_arr = list(collection.values())

        total_tree = combine_trees('*', TreeNode(monomial_arr[0][1]), monomial_arr[0][0])

        for tree, coefficient in monomial_arr[1:]:
            monomial = combine_trees('*', TreeNode(coefficient), tree)
            total_tree = combine_trees('+', total_tree, monomial)

        return total_tree

    # requires expansion first
    def collect(self):
        COUNTER = 1
        collection = {}

        def traverse(node):
            if node.value == '+':
                traverse(node.left)
                traverse(node.right)

            if node.value == '*':
                id = tuple(sorted(str(node).split()))

                if id in collection:
                    collection[id][COUNTER] += 1
                else:
                    collection[id] = [node, 1]

        traverse(self)

        return self.combine_monomials_coefficients(collection)

    # simple rules to simplify (currently not used in dataset generation)
    def simplify(self):
        if self.left:
            self.left = self.left.simplify_infix()
        if self.right:
            self.right = self.right.simplify_infix()

        if self.value in ['+', '-']:
            if self.left.value == self.right.value:
                if self.value == '+':
                    return TreeNode(f"2*{self.left.value}").simplify()
                elif self.value == '-':
                    return TreeNode('0')
        elif self.value == '*':
            if self.left.value == self.right.value:
                return TreeNode(f"{self.left.value}^2").simplify()
        return self

# generates a single case
def generate_case(TOTAL_VARIABLES_TRUE = ['x', 'y', 'z'],
                  TOTAL_VARIABLES_SUB = ['a', 'b', 'c'],
                  original = None,
                  expand=False,
                  collect=False,
                  substitute_max_depth = 3):

    if original :
        expression = TreeNode(original[0],fixed=original)
        original_expression = copy.deepcopy(expression)
    else :
        expression = TreeNode.generate_expression(3, TOTAL_VARIABLES_TRUE)
        original_expression = copy.deepcopy(expression)

    sub_variables = {}
    sub_string = ""

    for target in TOTAL_VARIABLES_TRUE:
        sub_variables[target] = TreeNode.generate_expression(substitute_max_depth, TOTAL_VARIABLES_SUB)
        sub_string += f"{target} : {sub_variables[target]}, "

    expression.substitute_variable(sub_variables)

    if collect :
        expression = expression.expansion().collect()
    elif expand :
        expression = expression.expansion()

    return [str(expression), sub_string[:-2], str(original_expression)]

# generates total number of cases
def generate_dataset(num_of_case, TOTAL_VARIABLES_TRUE = set(['x', 'y', 'z']),
                      TOTAL_VARIABLES_SUB = set(['a', 'b', 'c'])):
    training_cases = {"input_texts" : [], "target_texts": [], "original_texts" : []}

    for _ in range(num_of_case):
        training_case = generate_case(TOTAL_VARIABLES_TRUE, TOTAL_VARIABLES_SUB)
        training_cases["input_texts"].append(training_case[0])
        training_cases["target_texts"].append(training_case[1])
        training_cases["original_texts"].append(training_case[2])

    return training_cases

if __name__ == "__main__":
    # print(generate_case(['x','y'],['a','b','c'],original = '+*xx*yy',substitute_max_depth=2,expand=True))
    # print(generate_case(['x','y'],['a','b','c'],original = '+^x2^y2'))

    print(f"Before Expansion Form (Infix Notation): {TreeNode('*', fixed='*+ab+ab').stringify_infix()}")
    print(f"After Expansion Form (Infix Notation): {TreeNode('*', fixed='*+ab+ab').expansion().stringify_infix()}")
    print(f"After Collection Form (Infix Notation): {TreeNode('*', fixed='*+ab+ab').expansion().collect().stringify_infix()}")

    # EDGE CASE FOR LARGER EXPANSION
    # Collection should work after full expansion
    print(f"Before Expansion Form (Infix Notation): {TreeNode('*', fixed='*++abc++abc').stringify_infix()}")
    print(f"After Expansion Form (Infix Notation): {TreeNode('*', fixed='*++abc++abc').expansion().stringify_infix()}")
    print(f"After Collection Form (Infix Notation): {TreeNode('*', fixed='*++abc++abc').expansion().collect().stringify_infix()}")
