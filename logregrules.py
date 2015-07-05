import warnings
import Orange
import numpy as np
from orngABML import *

class RuleToAtt:
    """ Coverting Orange rule to an attribute """
    def __init__(self, attribute, rule, positive_value):
        self.attribute = attribute
        self.rule = rule
        self.positive_value = positive_value

    def __call__(self, example, returnType):
        if self.rule(example):
            return Orange.data.Value(self.attribute, self.positive_value)
        return Orange.data.Value(self.attribute, 0)

class LogRegFitter:
    """ Class learns a logistic regression model with L2 penalty. It uses gradient descent. An attribute can be a normal
    attribute, which is used as usual or a rule-based attribute that needs special handling during estimation of
    derivatives.
    """

    def __init__(self, data, original_data, rules, C, eps, weps, alpha):
        """
        Init accepts n vectors (corresponding to attributes) with rules attached to each attribute. IF rule is None,
        then attributes is just a normal attribute, otherwise conside difference between rule quality and
        relative frequency of examples in rule.

        C value and eps and alpha are parameters guiding optimization
        """
        self.data = data
        self.original_data = original_data
        self.rules = rules # 
        self.C = C
        self.eps = eps
        self.weps = weps
        self.alpha = alpha
        
        self.add_rules()
        self.initialize_values()
        self.learn_model()

    def add_rules(self):
        # add rules
        # compute penalties
        self.penalty = [np.array([1] * len(self.data)) for a in self.data.domain.attributes]
        self.rule_classes = [-1] * len(self.data.domain.attributes)
        new_attributes = [a for a in self.data.domain.attributes]
        if not self.rules:
            return
        for r in self.rules:
            newAttr = Orange.feature.Continuous(Orange.classification.rules.rule_to_string(r))
            newAttr.getValueFrom = RuleToAtt(newAttr, r, 1)
            new_attributes.append(newAttr)
            rf = r.classDistribution[int(r.classifier.defaultVal)] / r.classDistribution.abs
            penalty = r.quality / rf
            penalArray = [penalty if r(e) else 1 for e in self.original_data]
            self.penalty.append(np.array(penalArray))
            self.rule_classes.append(int(r.classifier.defaultVal))
        new_domain = Orange.data.Domain(new_attributes, self.data.domain.classVar)
        new_domain.add_metas(self.data.domain.get_metas())
        self.data = Orange.data.Table(new_domain, self.original_data)

    def initialize_values(self):
        # impute missing values with 0
        for ci in self.data:
            for at in self.data.domain.attributes:
                if ci[at].isSpecial():
                    ci[at] = 0

        self.x, self.ya = self.data.toNumpy("1a/c")
        if len(data.domain.classVar.values) == 2:
            self.y = [self.ya]
        else:
            self.y = []
            for vi, v in enumerate(data.domain.classVar.values):
                tmp = np.array(self.ya)
                tmp[tmp==vi] = -1
                tmp[tmp!=-1] = 0
                tmp[tmp==-1] = 1
                self.y.append(tmp)
        self.xy = [self.x * y[:, np.newaxis] for y in self.y]
        self.penalty = [1] + self.penalty
        self.xydiff = [self.x * (1.0 - penal[:, np.newaxis] for penal in self.penalty] 
        self.rule_classes = [-1] + self.rule_classes
        self.weights = [np.zeros(len(self.data.domain.attributes)+1) for y in self.y]
        self.z = [0] * len(self.weights)
        self.h = [0] * len(self.weights)
        self.minvals = np.array([-20] * len(self.data))
        self.maxvals = np.array([20] * len(self.data))
        #print self.penalty

    def err(self, i):
        error = np.dot(self.weights[i], self.weights[i])
        self.z[i] = np.dot(self.x, self.weights[i])
        self.z[i] = np.maximum(self.z[i], self.minvals)
        self.z[i] = np.minimum(self.z[i], self.maxvals)
        self.h[i] = 1.0 / (1 + np.exp(-self.z[i]))
        self.h[i][self.h[i]<1e-6] = 1e-6
        error += self.C * np.sum(-self.y[i] * np.log(self.h[i]) - (1 - self.y[i]) * np.log(1 - self.h[i]))
        return error

    def deriv(self, i):
        # ideally this product (xy * penalty) would be a product of vectors
        # 
        #

        return self.weights[i] + self.C * np.sum(self.x * self.h[i][:,np.newaxis] - 
                                                 self.xy[i] * self.penalty[i], axis=0)

    def learn_model(self):
        for i in range(len(self.weights)):
            new_error = self.err(i)
            old_error = 2 * new_error
            diff, old_diff = 1, 1e+10
            alpha = self.alpha
            # while new error is less than the old error or
            # difference between succeeding weights is larger than weps, repeat optimization
            #while new_error < old_error - self.eps or alpha > 0.0001:
            while old_diff > diff or alpha > 0.0001:
                #if new_error > old_error - self.eps:
                if old_diff <= diff:
                    alpha /= 2
                if diff < self.weps:
                    break
                old_weights = np.array(self.weights[i])
                self.weights[i] -= alpha * self.deriv(i)
                old_diff = diff
                diff = np.linalg.norm(old_weights - self.weights[i])
                old_error, new_error = new_error, self.err(i)
        print self.weights

    def __call__(self, example):
        example = Orange.data.Instance(self.data.domain, example)
        for at in example.domain.attributes:
            if example[at].isSpecial():
                example[at] = 0

        examples = Orange.data.Table(self.data.domain)
        examples.append(example)

        x = examples.toNumpy("1a")
        x = x[0][0]

        dist = []
        for wi, w in enumerate(self.weights):
            z = sum(x * w)
            dist.append(1.0 / (1.0 + np.exp(-z)))
        if len(dist) == 1:
            # two classes case
            dist = Orange.statistics.distribution.Discrete([1.0-dist[0], dist[0]])
            dist.variable = example.domain.classVar 
        else:
            sd = sum(dist)
            dist = [d/sd for d in dist]
            dist = Orange.statistics.distribution.Discrete(dist)
            dist.variable = example.domain.classVar 
        return dist

    
class LogRegRules(Orange.classification.Learner):
    def __init__(self, use_rules = False, m = 2, **kwargs):
        #self.rule_learner = Orange.classification.rules.ABCN2(argument_id="Arguments", att_sig = 0.5, m = m, debug=True, **kwargs)
        self.rule_learner = Orange.classification.rules.ABCN2(att_sig = 0.5, m = m, **kwargs)
        self.rule_learner.classifier = Orange.classification.rules.RuleClassifier_bestRule
        self.use_rules = use_rules
        self.m = m

    def __call__(self, instances, weightID=0):
        continuizer = Orange.data.continuization.DomainContinuizer()
        continuizer.multinomial_treatment = continuizer.FrequentIsBase
        continuizer.continuous_treatment = continuizer.NormalizeByVariance
        continuizer.class_treatment = continuizer.LeaveUnlessTarget
        cont_domain = continuizer(instances)
        cont_instances = Orange.data.Table(cont_domain, instances)
        
        if self.use_rules:
            rules = self.rule_learner(instances).rules
            fitter = LogRegFitter(cont_instances, instances, rules, 1.0, 0.001, 0.001, 0.01)
        else:
            fitter = LogRegFitter(cont_instances, instances, None, 1.0, 0.001, 0.001, 0.01)


        return LogRegRulesClassifier(fitter, None)


        # prepare new instances
        # binarization
        # normalization
        self.prior = Orange.statistics.distribution.Distribution(instances.domain.class_var, instances)
        
        continuizer = Orange.data.continuization.DomainContinuizer()
        continuizer.multinomial_treatment = continuizer.FrequentIsBase
        continuizer.continuous_treatment = continuizer.NormalizeByVariance
        continuizer.class_treatment = continuizer.LeaveUnlessTarget
        cont_domain = continuizer(instances)
        cont_instances = Orange.data.Table(cont_domain, instances)

        # tune logistic regression and learn on original domain
        lr = Orange.classification.logreg.LibLinearLogRegLearner(C=1, eps=0.01, normalization=False, bias=1)
        tuner = Orange.tuning.Tune1Parameter(learner=lr,
                                   parameter="C",
                                   #values=[0.001, 0.01,  0.1, 1.0, 10., 100.],
                                   values=[1000.001],
                                   evaluate=Orange.evaluation.scoring.AUC, verbose=0, return_what = Orange.tuning.TuneParameters.RETURN_LEARNER)
        # first build a l ogistic model
        learner = tuner(cont_instances, weightID)
        model = learner(cont_instances, weightID)
        # learn rules (consider self.max_rules)
        if self.use_rules:
            # create a collection of extended attribute
            #new_attributes = [a for a in cont_domain.attributes]
            #minmax = [(0, 0, None) for a in cont_domain.attributes]
            #coeff = []#[(0, None) for a in cont_domain.attributes]
            # until there are added rules --> relearn
            rules = self.rule_learner(instances).rules
            print "all rules"
            for r in rules:
                print Orange.classification.rules.rule_to_string(r)
            accepted_rules, accepted_set = [], set()
            #for r in rules:
            #    diff = eval_rule(model, r)
            #    if diff > 0:
            #        accepted_rules.append((diff, r))
            found_rule = True
            while found_rule:
                # learn a model from current rules
                new_attributes = [a for a in cont_domain.attributes]
                for pos, r in accepted_rules:
                    newAttr = Orange.feature.Continuous(Orange.classification.rules.rule_to_string(r))
                    newAttr.getValueFrom = RuleToAtt(newAttr, r, pos)
                    new_attributes.append(newAttr)
                new_domain = Orange.data.Domain(new_attributes, cont_domain.classVar)
                new_domain.add_metas(cont_domain.get_metas())
                new_instances = Orange.data.Table(new_domain, instances)
                new_model = learner(new_instances, weightID)
                print "weights", new_model.weights 
                # find best rule
                max_err, best_dz = 0.0, 0
                sel_rule = None
                for r in rules:
                    if Orange.classification.rules.rule_to_string(r) in accepted_set:
                        continue
                    diff = eval_rule(new_model, r)
                    if diff > 0:
                        dz, derr = findz(new_model, r)
                        if derr > max_err:
                            sel_rule = r
                            max_err = derr
                            best_dz = dz
                if not sel_rule:
                    break

                #if len(accepted_rules) > 0:
                #break

                print "selected rule: ", Orange.classification.rules.rule_to_string(sel_rule), max_err

                # now run bisection to achieve with rule goal_z
                # goal_z = findzdiff(new_model, sel_rule)
                best_pos = bisect(0.0, 1.0, GetZ(new_attributes, sel_rule, cont_domain, instances, weightID, learner), best_dz, 0.01) 
                print "best", best_dz, best_pos 

                accepted_rules.append((best_pos, sel_rule))
                #accepted_rules.append((0.02, sel_rule))
                accepted_set.add(Orange.classification.rules.rule_to_string(sel_rule))
                #print accepted_set


        else:
            return LogRegRulesClassifier(model, cont_domain, None)
    

class LogRegRulesClassifier(Orange.classification.Classifier):
    def __init__(self, model, rules):
        self.model = model
        self.rules = rules

    def __call__(self, instance, returnType = Orange.classification.Classifier.GetValue):
        probDist = self.model(instance)
        if returnType == Orange.classification.Classifier.GetProbabilities:
            return probDist
        elif returnType == Orange.classification.Classifier.GetValue:
            return probDist.modus()
        return probDist.modus(), probDist


datasets = [
"abalone",
"ailerons",
#"adult",
#"anneal",
#"audiology",
#"auto-mpg",
#"balance-scale",
"breast-cancer",
"breast-cancer-wisconsin-cont",
"bupa",
"car",
"cmc",
"coil2000test",
"connect-4",
"crx",
"dermatology",
"galaxy",
"german",
"glass",
"hayes-roth_learn",
"heart_disease",
"hepatitis",
"horse-colic",
"housing",
"imports-85",
"iris",
"ionosphere",
"leukemia",
"liver-disorders",
"lung",
"lung-cancer",
"lymphography",
"monks-1",
"monks-2",
"monks-3",
"pima-indians-diabetes",
"post-operative",
"promoters",
"prostate",
"primary-tumor",
"segmentation",
"servo",
"shuttle-landing-control",
"soybean-large-train",
"tic_tac_toe",
"titanic",
"vehicle",
"voting",
"vowel-train",
"wdbc",
"wine",
"yeast",
"zoo"
]

#datasets=["zoo", "breast-cancer", "pima-indians-diabetes"]

#datasets = ['pima-indians-diabetes']

if __name__ == '__main__':
    # test it here
    cl = LogRegRules()
    clWith = LogRegRules(use_rules = True)

    # run cross-validation
    learners = [cl, clWith]
    for ds in datasets:
        print " *** " * 10
        print ds
        data = Orange.data.Table("./domene/%s/%s.tab"%(ds,ds))
        if len(data.domain.classVar.values) > 2:
            print "Leaving this data set as it has more than 2 classes."
            continue
        if len(data) > 1000 or len(data.domain.attributes) > 1000:
            print "Leaving this data set as it is too large."
            continue
        print "dataset=%s, datalen=%d, n_classes=%d"%(ds, len(data), len(data.domain.classVar.values))

        cv = Orange.evaluation.testing.cross_validation(learners, data, folds=10)
        print "Results!!!!"
        print "CA: ", ["%.4f" % score for score in Orange.evaluation.scoring.CA(cv)]    
        print "AUC: ", ["%.4f" % score for score in Orange.evaluation.scoring.AUC(cv)]    
        print "Brier: ", ["%.4f" % score for score in Orange.evaluation.scoring.Brier_score(cv)]    





    
