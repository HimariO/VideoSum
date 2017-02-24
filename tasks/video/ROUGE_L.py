from nltk.corpus import wordnet as wn
import math

class ROUGE_L:
    def __init__(self,baise1,baise2,step,e):
        self.baise1=baise1
        self.baise2=baise2
        self.step=step
        if e>=0.1:
            self.e=0.1
        else:
            self.e=e
        self.t=0.0001
        return 
    def set_inputs(self,s1):
        self.inputs=s1.split(" ")
    def set_targes(self,lis):
        self.targes=[]
        lis_len=len(lis)
        for i in range(lis_len):
            self.targes.append(lis[i].split(" "))
    def myactive_function(self,cost):
        if cost>=(1-self.t**2):
            return (1-cost)
        x=math.pi*(2*cost-1)
        outloss=1/(1+math.exp(-1*x))
        flag=50
        if outloss>=self.baise1:
            flag=100*self.e
            outloss=1-(self.e)**((1-outloss)*self.step)
        elif outloss<self.baise2:
            flag=10/self.e
            outloss=(self.e)**((self.baise2-outloss)*self.step)
        outloss=(1-outloss)*flag
        if outloss<=self.t:
            outloss=0
        return outloss

    def mycross_entropy(self,cost):
 #       outloss=-1*math.log(self.myactive_function(cost))
        return self.myactive_function(cost)

    def similar_score(self,s1,s2):
        w1=wn.synsets(s1)
        w2=wn.synsets(s2)
        if len(w1)==0 or len(w2)==0:
            if s1==s2:
                return 1
            else:
                return 0
        out=w1[0].path_similarity(w2[0])
        return out

    def LCS(self,index_targes):
        s1=self.inputs
        s2=self.targes[index_targes]
        baseline=2/math.pi
        m=len(s1)+1
        n=len(s2)+1
        record=[[0 for i in range(n)] for j in range(m)]
        for i in range(1,m):
            for j in range(1,n):
                word_score=self.similar_score(s1[i-1],s2[j-1])
                if word_score!=None and word_score>=baseline:
                    record[i][j] = record[i-1][j-1]+word_score
                elif record[i-1][j] >= record[i][j-1]:
                    record[i][j] = record[i-1][j]
                else:
                    record[i][j] = record[i][j-1]
        return record[m-1][n-1]

    def Caculate_max_LCS(self):
        if len(self.targes)<=0:
            return -1
        if len(self.inputs)<=0:
            return -1
        targes_len=len(self.targes)
        max_p=0
        max_q=0
        index_targes=0
        for i in range(targes_len):
            if len(self.targes[i])>0:
                LCS_score=self.LCS(i)
                if max_q<LCS_score:
                    max_q=LCS_score
                if max_p<(LCS_score/len(self.targes[i])):
                    max_p=LCS_score/len(self.targes[i])
        return max_p,(max_q/len(self.inputs))
    def Caculate_ROUGE_L(self):
        max_p,max_q=self.Caculate_max_LCS()
        if max_p==0 or max_q==0:
            return 1000
        alpha=1.2
        if max_p>max_q:
            F=(1+alpha**2)*max_p*max_q/(max_p+(alpha**2)*max_q)
        else:
            F=(1+alpha**2)*max_p*max_q/(max_q+(alpha**2)*max_p)
        return self.mycross_entropy(F)

