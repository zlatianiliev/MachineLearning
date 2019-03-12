# Bayes Theorem

## Formula: P(A|B) = P(B|A) * P(A) / P(B), where P - probability, A - event, B - event

## In words:

## The probability of a hypothesis A, conditional on a new piece of evidence E is equal to
## the probability of the evidence given the hypothesis times the prior probability of the hypothesis
## over the prior probability of the evidence

## The probability of observing event A if B is true is equal to
## the probability of observing event B if A is true times the probability of A
## over the probability of B

## Example: Given the test for brest cancer is 90% accurate when you have cancer and gives 10% false
## positives. If 1% of women between age of 45-50 have cancer, what is the probability that a random
## woman takes the test and tests poistive actually has cancer?

## P(Cancer|+) = P(+|Cancer) * P(Cancer) / P(+) = 0.9 * 0.01 / (0.01 * 0.9 + 0.99 * 0.1) = 8.33%

## Three representation
            *
Cancer 1% /   \ 99% No Cancer
         *     \
    90% / \ 10% \
       +   -     \
                  *
             10  / \  90%
                +   -