import csv
import itertools
import sys

debug = False

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    def transitionModel(motherGenes, fatherGenes, childGenes):
        """
        return
        :param parentGenes: number of gene copies in parent (0, 1 or 2)
        :param childGenes: number of gene copies in child (0, 1, 2)
        :return: P(childGenes | motherGenes ^ fatherGenes)
        """
        if debug:
            print(f"motherGenes: {motherGenes}, fatherGenes: {fatherGenes}, child has {childGenes} genes")
        if motherGenes == 0:
            motherModel = {'Arm1Pos': PROBS["mutation"], 'Arm1Neg': 1-PROBS["mutation"], 'Arm2Pos': PROBS["mutation"], 'Arm2Neg': 1-PROBS["mutation"]}
        elif motherGenes == 2:
            motherModel = {'Arm1Pos': 1-PROBS["mutation"], 'Arm1Neg': PROBS["mutation"], 'Arm2Pos': 1-PROBS["mutation"], 'Arm2Neg': PROBS["mutation"]}
        elif motherGenes == 1:
            motherModel = {'Arm1Pos': PROBS["mutation"], 'Arm1Neg': 1-PROBS["mutation"], 'Arm2Pos': 1-PROBS["mutation"], 'Arm2Neg': PROBS["mutation"]}

        if fatherGenes == 0:
            fatherModel = {'Arm1Pos': PROBS["mutation"], 'Arm1Neg': 1 - PROBS["mutation"], 'Arm2Pos': PROBS["mutation"],
                           'Arm2Neg': 1 - PROBS["mutation"]}
        elif fatherGenes == 2:
            fatherModel = {'Arm1Pos': 1 - PROBS["mutation"], 'Arm1Neg': PROBS["mutation"], 'Arm2Pos': 1 - PROBS["mutation"],
                           'Arm2Neg': PROBS["mutation"]}
        elif fatherGenes == 1:
            fatherModel = {'Arm1Pos': PROBS["mutation"], 'Arm1Neg': 1 - PROBS["mutation"], 'Arm2Pos': 1 - PROBS["mutation"],
                           'Arm2Neg': PROBS["mutation"]}

        probValue0 = (fatherModel['Arm1Neg'] * motherModel['Arm1Neg'] + fatherModel['Arm2Neg']*motherModel['Arm2Neg'] +
                     fatherModel['Arm1Neg'] * motherModel['Arm2Neg'] + fatherModel['Arm2Neg'] * motherModel['Arm1Neg'])
        probValue1 = (fatherModel['Arm1Pos'] * motherModel['Arm1Neg'] + fatherModel['Arm1Pos']*motherModel['Arm2Neg'] +
                     fatherModel['Arm2Pos'] * motherModel['Arm1Neg'] + fatherModel['Arm2Pos'] * motherModel['Arm2Neg'] +
                     fatherModel['Arm1Neg'] * motherModel['Arm1Pos'] + fatherModel['Arm1Neg'] * motherModel['Arm2Pos'] +
                     fatherModel['Arm2Neg'] * motherModel['Arm1Pos'] + fatherModel['Arm2Neg'] * motherModel['Arm2Pos'] )
        probValue2 = (fatherModel['Arm1Pos'] * motherModel['Arm1Pos'] + fatherModel['Arm1Pos']*motherModel['Arm2Pos'] +
                     fatherModel['Arm2Pos'] * motherModel['Arm1Pos'] + fatherModel['Arm2Pos'] * motherModel['Arm2Pos'])
        probSum = probValue2 + probValue1 + probValue0
        probValue2 /= probSum
        probValue1 /= probSum
        probValue0 /= probSum

        #return child probability given number of genes he has:
        if childGenes == 0:
            return probValue0
        elif childGenes == 1:
            return probValue1
        elif childGenes == 2:
            return probValue2
        #end of transitionModel


    if debug:
        for aPerson in people:
            print(f"{aPerson} -- {people[aPerson]}")
        print(f"one gene: {one_gene}")
        print(f"two genes: {two_genes}")
        print(f"have trait: {have_trait}")

    names = set(people)
    noGene = names - one_gene - two_genes
    noTrait = names - have_trait

    if debug:
        print(f"People with noGene: {noGene}")
        print(f"People with no trait: {noTrait}")

    query = dict()
    for aPerson in names:
        nGenes = 0
        trait = False
        if aPerson in one_gene:  nGenes = 1
        if aPerson in two_genes:  nGenes = 2
        if aPerson in have_trait:
            trait = True
        query[aPerson] = {"genes": nGenes, "trait": trait}
        if debug: print(f"query[{aPerson}]: {query[aPerson]}")

    #compute joint probability which will be a multiplication of all persons in names
    result = 1
    for aPerson in names:
        nGenes = query[aPerson]['genes']
        hasTrait = query[aPerson]['trait']

        #test if no parents
        if people[aPerson]['mother'] == None:
            if debug: print(f"{aPerson} has no parent")
            geneProb = PROBS["gene"][nGenes]
            if debug: print(f"PROBS['gene'][{nGenes}] = {geneProb:.5f}")

        #handle if parents:
        else:
            mother = people[aPerson]['mother']
            motherGenes = 0
            if mother in one_gene:
                motherGenes = 1
            elif mother in two_genes:
                motherGenes =2
            father = people[aPerson]['father']
            fatherGenes = 0
            if father in one_gene:
                fatherGenes = 1
            elif father in two_genes:
                fatherGenes = 2
            geneProb = transitionModel(motherGenes= motherGenes, fatherGenes= fatherGenes, childGenes= nGenes)
        #end handle if parents

        traitProb = PROBS["trait"][nGenes][hasTrait]
        if debug: print(f"PROBS['trait'][{nGenes}][{hasTrait}] = {traitProb:.5f}")

        personProb = geneProb * traitProb
        if debug: print(f"Probability that {aPerson} has {nGenes} genes and trait {hasTrait} is {personProb:.5f}")

        result *= personProb
        if debug: print(f"overall result is now: {result:.5f}")
    #end of loop on all names

    if debug: print(f"loop ended. Returning result: {result:.5f}")
    return result



def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    names = set(probabilities)
    noGene = names - one_gene - two_genes
    noTrait = names - have_trait

    if debug:
        print(f"People with noGene: {noGene}")
        print(f"People with no trait: {noTrait}")

    for aPerson in names:
        nGenes = 0
        trait = False
        if aPerson in one_gene:  nGenes = 1
        if aPerson in two_genes:  nGenes = 2
        if aPerson in have_trait: trait = True
        if debug: print(f"Updating {aPerson} with {nGenes} gene copies and trait: {trait} with probability {p:.6f}")

        probabilities[aPerson]['gene'][nGenes] += p
        if trait == True:
            if debug: print(f"Updating {aPerson} with {nGenes} gene copies and trait: {trait} with probability {p:.6f}")

            probabilities[aPerson]['trait'][True] += p
        else: probabilities[aPerson]['trait'][False] += p
    return



def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """

    for aPerson in probabilities:
        sum = 0
        for i in range(0, 3):
            sum += probabilities[aPerson]['gene'][i]
        if debug: print(f"sum for {aPerson} is {sum:.5f}")
        newSum = 0
        for i in range(0,3):
            probabilities[aPerson]['gene'][i] /= sum
            newSum += probabilities[aPerson]['gene'][i]
            if debug: print(f"new probability: {probabilities[aPerson]['gene'][i]}")
        if debug: print(f"New sum: {newSum:.5f}")

        sumTrait = probabilities[aPerson]['trait'][True] + probabilities[aPerson]['trait'][False]
        if debug:
            print(f"sum of traits for {aPerson} is: {sumTrait} with True= {probabilities[aPerson]['trait'][True]} and False {probabilities[aPerson]['trait'][False]}")
        probabilities[aPerson]['trait'][True] /= sumTrait
        probabilities[aPerson]['trait'][False] /= sumTrait
        if debug:
            print(f"Prob distribution for {aPerson} is: True= {probabilities[aPerson]['trait'][True]} and False {probabilities[aPerson]['trait'][False]}")


    return



if __name__ == "__main__":
    main()
