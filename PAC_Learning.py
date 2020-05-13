import math


def main():
    PACBound1 = lambda genericErrorRate, sampleFailureRate, VC: (1 / genericErrorRate) * (4 * math.log2(2 / sampleFailureRate) + 8 * VC * math.log2(13 / genericErrorRate))
    PACBound2 = lambda trainingErrorRate, sampleFailureRate, sampleNumbers, VC: (1 / sampleNumbers) * (trainingErrorRate * 4 * math.log(4 / sampleFailureRate, math.e) + 4 * VC * math.log(2*math.e*sampleNumbers / VC, math.e))
    PACBound3 = lambda sampleFailureRate, sampleNumbers, VC: max((VC-1) / (32*sampleNumbers), (1 / sampleNumbers) * math.log(1 / sampleFailureRate, math.e))

    print("Required Number of Samples:", PACBound1(0.07, 0.05, 9))
    print("The Upper Bound of the Generic Error:", PACBound2(0.27, 0.05, 3700, 9))
    print("The Lower Bound of the Generic Error:", PACBound3(0.05, 3700, 9))

    return


if __name__ == '__main__':
    main()
