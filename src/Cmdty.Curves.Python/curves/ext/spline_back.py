import scipy as sp
import numpy as np
from datetime import timedelta
from math import pow
from scipy.sparse import csr_matrix


def delta_pow(time_to_start, time_to_end, power):
    return pow(time_to_end, power) - pow(time_to_start, power)


def enumerate_hours(start_date, end_date):
    for n in range(int((end_date - start_date)/timedelta(hours=1) + 1)):
        yield start_date + timedelta(hours=n)


def create_2h_bottom_right_submatrix(contract, curve_start, time_func):
    "TODO: contract is a Contract object, time_func is a func of {T,T, double}"
    timeToStart = time_func(curve_start, contract['Start'])
    timeToEnd = time_func(curve_start, contract['End']+timedelta(hours=1))

    subMatrix = csr_matrix((3,3), dtype=float).todense()

    deltaPow2 = delta_pow(timeToStart, timeToEnd, 2.0)
    deltaPow3 = delta_pow(timeToStart, timeToEnd, 3.0)
    deltaPow4 = delta_pow(timeToStart, timeToEnd, 4.0)

    subMatrix[0, 0] = 8.0 * (timeToEnd - timeToStart)
    subMatrix[0, 1] = 12.0 * deltaPow2
    subMatrix[0, 2] = 16.0 * deltaPow3

    subMatrix[1, 0] = 12.0 * deltaPow2
    subMatrix[1, 1] = 24.0 * deltaPow3
    subMatrix[1, 2] = 36.0 * deltaPow4

    subMatrix[2, 0] = 16.0 * deltaPow3
    subMatrix[2, 1] = 36.0 * deltaPow4
    subMatrix[2, 2] = 57.6 * delta_pow(timeToStart, timeToEnd, 5.0)

    return subMatrix


def build(contracts, weighting, mult_adjust_func, add_adjust_func, 
          time_func, front_first_derivative=None, back_first_derivative=None):
    #TODO: contracts = list(contract objects), weighting = Func<T, dbl>
    if len(contracts) < 2:
        raise ValueError("contracts must have at least two elements", nameof(contracts))

    curveStartPeriod = contracts[0]['Start']
    numGaps = 0
    timeToPolynomialBoundaries = []

    #TODO: optionally do/don't allow gaps in contracts
    for i in range(len(contracts) - 1):
        contractEnd = contracts[i]['End']
        nextContractStart = contracts[i + 1]['Start']

        if contractEnd >= nextContractStart:
            raise Exception('Contracts are overlapping')
        timeToPolynomialBoundaries.append(time_func(curveStartPeriod, nextContractStart))

        if ((contractEnd - nextContractStart) / timedelta(hours=1)) < -1:
            numGaps += 1
            timeToPolynomialBoundaries.append(
                time_func(curveStartPeriod, contractEnd.next()))
    
    numPolynomials = len(contracts) + numGaps
    numCoefficientsToSolve = numPolynomials * 5

    numConstraints = (numPolynomials - 1) * 3 + numPolynomials - numGaps + (
        0 if front_first_derivative is None else 1) + (0 if back_first_derivative is None else 1)

    #matrixbuilder TODO: not used
    #vectorbuilder 

    constraintMatrix = csr_matrix(
        (numConstraints, numCoefficientsToSolve), dtype=float).todense()
    vector = csr_matrix(
        (numPolynomials * 5 + numConstraints, 1), dtype=float).todense()

    twoHMatrix = csr_matrix((numPolynomials * 5, numPolynomials * 5), dtype=float).todense()

    inputContractIndex = 0
    gapFilled = False

    rowNum = 0
    for i in range(numPolynomials):
        colOffset = i * 5
        if i < numPolynomials - 1:
                
            timeToPolynomialBoundary = timeToPolynomialBoundaries[i]
            timeToPolynomialBoundaryPow2 = pow(timeToPolynomialBoundary, 2)
            timeToPolynomialBoundaryPow3 = pow(timeToPolynomialBoundary, 3)
            timeToPolynomialBoundaryPow4 = pow(timeToPolynomialBoundary, 4)

            #Polynomial equality at boundaries
            constraintMatrix[rowNum, colOffset] = 1.0
            constraintMatrix[rowNum, colOffset + 1] = timeToPolynomialBoundary
            constraintMatrix[rowNum, colOffset + 2] = timeToPolynomialBoundaryPow2
            constraintMatrix[rowNum, colOffset + 3] = timeToPolynomialBoundaryPow3
            constraintMatrix[rowNum, colOffset + 4] = timeToPolynomialBoundaryPow4

            constraintMatrix[rowNum, colOffset + 5] = -1.0
            constraintMatrix[rowNum, colOffset + 6] = -timeToPolynomialBoundary
            constraintMatrix[rowNum, colOffset + 7] = -timeToPolynomialBoundaryPow2
            constraintMatrix[rowNum, colOffset + 8] = -timeToPolynomialBoundaryPow3
            constraintMatrix[rowNum, colOffset + 9] = -timeToPolynomialBoundaryPow4

            #Polynomial first derivative equality at boundaries
            constraintMatrix[rowNum + 1, colOffset] = 0.0
            constraintMatrix[rowNum + 1, colOffset + 1] = 1.0
            constraintMatrix[rowNum + 1, colOffset + 2] = 2.0 * timeToPolynomialBoundary
            constraintMatrix[rowNum + 1, colOffset + 3] = 3.0 * timeToPolynomialBoundaryPow2
            constraintMatrix[rowNum + 1, colOffset + 4] = 4.0 * timeToPolynomialBoundaryPow3

            constraintMatrix[rowNum + 1, colOffset + 5] = 0.0
            constraintMatrix[rowNum + 1, colOffset + 6] = -1.0
            constraintMatrix[rowNum + 1, colOffset + 7] = -2.0 * timeToPolynomialBoundary
            constraintMatrix[rowNum + 1, colOffset + 8] = -3.0 * timeToPolynomialBoundaryPow2
            constraintMatrix[rowNum + 1, colOffset + 9] = -4.0 * timeToPolynomialBoundaryPow3

            #Polynomial second derivative equality at boundaries
            constraintMatrix[rowNum + 2, colOffset] = 0.0
            constraintMatrix[rowNum + 2, colOffset + 1] = 0.0
            constraintMatrix[rowNum + 2, colOffset + 2] = 2.0
            constraintMatrix[rowNum + 2, colOffset + 3] = 6.0 * timeToPolynomialBoundary
            constraintMatrix[rowNum + 2, colOffset + 4] = 12.0 * timeToPolynomialBoundaryPow2

            constraintMatrix[rowNum + 2, colOffset + 5] = 0.0
            constraintMatrix[rowNum + 2, colOffset + 6] = 0.0
            constraintMatrix[rowNum + 2, colOffset + 7] = -2.0
            constraintMatrix[rowNum + 2, colOffset + 8] = -6 * timeToPolynomialBoundary
            constraintMatrix[rowNum + 2, colOffset + 9] = -12.0 * timeToPolynomialBoundaryPow2

        #Contract price constraint
        if i==0 or ((contracts[inputContractIndex - 1]['End'] - contracts[inputContractIndex]['Start']) / timedelta(hours=1) == -1) or gapFilled:
            contract = contracts[inputContractIndex]
            sumWeight = 0.0
            sumWeightMult = 0.0
            sumWeightMultTime = 0.0
            sumWeightMultTimePow2 = 0.0
            sumWeightMultTimePow3 = 0.0
            sumWeightMultTimePow4 = 0.0
            sumWeightMultAdd = 0.0

            iterated = list(enumerate_hours(contract['Start'], contract['End']))
            for timePeriod in iterated:
                timeToPeriod = time_func(curveStartPeriod, timePeriod)
                weight = weighting(timePeriod)
                multAdjust = mult_adjust_func(timePeriod)
                addAdjust = add_adjust_func(timePeriod)

                sumWeight += weight
                sumWeightMult += weight * multAdjust
                sumWeightMultTime += weight * multAdjust * timeToPeriod
                sumWeightMultTimePow2 += weight * multAdjust * pow(timeToPeriod, 2.0)
                sumWeightMultTimePow3 += weight * multAdjust * pow(timeToPeriod, 3.0)
                sumWeightMultTimePow4 += weight * multAdjust * pow(timeToPeriod, 4.0)
                sumWeightMultAdd += weight * multAdjust * addAdjust

            priceConstraintRow = rowNum if i==(numPolynomials - 1) else rowNum + 3

            #Coefficient of a
            constraintMatrix[priceConstraintRow, colOffset] = sumWeightMult
            #Coefficient of b
            constraintMatrix[priceConstraintRow, colOffset + 1] = sumWeightMultTime
            #Coefficient of c
            constraintMatrix[priceConstraintRow, colOffset + 2] = sumWeightMultTimePow2
            #Coefficient of d
            constraintMatrix[priceConstraintRow, colOffset + 3] = sumWeightMultTimePow3
            #Coefficient of e
            constraintMatrix[priceConstraintRow, colOffset + 4] = sumWeightMultTimePow4
            
            vector[numPolynomials * 5 + priceConstraintRow] = sumWeight * contract['Price'] - sumWeightMultAdd
            
            subMatrix = create_2h_bottom_right_submatrix(
                contract, curveStartPeriod, time_func)
            startIndex = i*5+2
            endIndexRow = subMatrix.shape[0]
            endIndexCol = subMatrix.shape[1]
            twoHMatrix[startIndex:endIndexRow,
                       startIndex:endIndexCol] = subMatrix
            
            inputContractIndex += 1
            rowNum += 4
            gapFilled = False
        else:
            #Gap in contracts
            rowNum += 3
            gapFilled = True

    return ""

