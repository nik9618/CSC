/*
todo :
1. optimize z coding.
2. multithreads.


in doubt ? 
1. normalize input ? 
*/

#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include "omp.h"
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <cfloat>

using namespace std;

#define d(j,k,i) 	D[j][k][i+dsize]
#define div(j,k,i) 	D[j][k][i+dsize]
#define s(j,i)		S[j][i]
#define z(j,k,i) 	Z[j][k][i+dsize]
#define zn(j,k,i) 	ZN[j][k][i+dsize]
#define b(j,k,i) 	B[j][k][i+dsize]
#define nd(j,k,i) 	ND[j][k][i+dsize]

#define T 30
#define LAMBDA 0.005
#define inferenceMinRound 100
#define inferencePercentBreak 0.2
#define inferenceMaxRound 1000
// #define inferenceDiffLossBreak 0.01

#define dictMinRound 250
#define dictMaxRound 500
#define dictDiffLossBreak 0.1
#define dictPercentBreak 0.1
#define dictstepsize 0.000000001

int * inD;
int * outD;

int in_dsize =0;
int in_k =0;
int in_sLen = 0;

int dsize = 7;
int ssize = 1214;
int K = 40;

vector< vector< vector<double> > >D; //done
vector< vector< vector<double> > >Z; //done
vector< vector< vector<double> > >ZN; //done
vector< vector< vector<double> > >B; //done
vector< vector<double> > S; //done
vector< vector< vector<double> > > Zn;
vector< vector< vector<double> > > ND;

vector< vector<double> > in_D;
vector<double> in_L;
vector< vector<double> > in_S;
vector< vector<short> > in_codek;
vector< vector<short> > in_codei;
vector< vector<double> > in_codeval;

vector<int> rd;


int parseBinary(int *dsize, int *k, int *sLen, vector< vector<double> > *D,vector<double> *l, vector< vector<double> > *s, vector< vector<short> > *ck, vector< vector<short> > *ci, vector< vector<double> > *cv)
{
	ifstream file;
	file.open("/home/kanit/anomalydeep/dataout_reg0.05_ds7_k50/data-0.bin",ifstream::binary);

	if (file.is_open()) {
	
		// parse Dict
		int recordLength = 0;
		int _dsize;
		int _k;

		file.read((char*)&recordLength, 4);
		file.read((char*)&_dsize, 4);
		file.read((char*)&_k, 4);

		*dsize = _dsize;
		*k = _k;

		vector< vector<double> > Dtmp = vector< vector<double> >(_k, vector<double>(_dsize*2+1,0));

		for(int i = 0 ; i < _k ; i++)
		{
			for (int j = 0 ; j < _dsize*2+1;j++)
			{
				float d = 0;
				file.read((char*)&d,4);
				Dtmp[i][j]=(double)d;
			}
		} 
		*D = Dtmp;

		// parse Signal
		int chunkLength;
		double loss;
		short sigLen;
		short orig_sigLen;
		short zLen;
		float ftmp;
		short stmp;
		short dindex;
		short sindex;
		char * buf = (char*)malloc(100000);
		
		int count = 0 ;
		while(file.good())
		{
			file.read((char*)&chunkLength, 4);
			if(!file.good()) break;
			file.read(buf, chunkLength);
			if(!file.good()) break;
			int idx = 0 ;
			int take = 0;
			take=8; memcpy(&loss,buf+idx,take); idx+=take;
			take=2; memcpy(&sigLen,buf+idx,take); idx+=take;
			orig_sigLen = sigLen;
			count++;
		}
		*sLen = orig_sigLen;
		
		vector<double> ls = vector<double>(count,0);
		vector< vector<double> > sig = vector< vector<double> >(count, vector<double>(sigLen,0) );
		vector< vector<short> > codek = vector< vector<short> >(count, vector<short>(0,0) );
		vector< vector<short> > codei = vector< vector<short> >(count, vector<short>(0,0) );
		vector< vector<double> > codeval = vector< vector<double> >(count, vector<double>(0,0) );
		file.clear();
		file.seekg(0, ios::beg);
		file.read((char*)&recordLength, 4);
		file.read(buf, recordLength);
		
		for(int i =0 ;  i < count ;i++)
		{
			file.read((char*)&chunkLength, 4);
			file.read(buf, chunkLength);	
			
			int idx = 0 ;
			int take = 0;

			take=8; memcpy(&loss,buf+idx,take); idx+=take;
			take=2; memcpy(&sigLen,buf+idx,take); idx+=take;
			
			if(orig_sigLen != sigLen) printf("SIGNAL LENGTH INCONSISTENCE\n");

			ls[i] = loss;

			for(int j=0 ; j<sigLen; j++)
			{	
				take=4; memcpy(&ftmp,buf+idx,take); idx+=take; 
				sig[i][j] = (double)ftmp;
			}

			take=2; memcpy(&zLen,buf+idx,take); idx+=take;
			
			codek[i].resize(zLen);
			codei[i].resize(zLen);
			codeval[i].resize(zLen);

			for(int j=0 ; j<zLen; j++)
			{	
				take=2; memcpy(&stmp,buf+idx,take); idx+=take; 
				codek[i][j]=stmp;
				take=2; memcpy(&stmp,buf+idx,take); idx+=take; 
				codei[i][j]=stmp;
				take=4; memcpy(&ftmp,buf+idx,take); idx+=take;
				codeval[i][j]=ftmp;
			}
		}

		*s = sig;
		*ck = codek;
		*ci = codei;
		*cv = codeval;
		*l = ls;

		return count;
	}
	else
	{
		return 0;
	}
}

vector<int> shuffleArray(int total)
{
	vector<int> v(total,0);
	for(int i = 0 ; i < total ; i++)
		v[i]=i;
	std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(v.begin(), v.end(), g);
    return v;
}

double genOldLoss(vector<double> S1,vector<double> S2)
{
	double ls=0;
	for(int i =0 ; i < S1.size();i++)
	{
		ls+= (S1[i]-S2[i])*(S1[i]-S2[i]);
	}
	return ls;
}

vector<double> genOldReconstruct(vector< vector<double> > D, vector<double> S, vector<short> ck, vector<short>ci, vector<double> cv)
{
	vector<double> recon(S.size(),0);
	vector< vector<double> > Z(D.size(), vector<double>(S.size()+2*D[0].size(),0)); 
	int ds = (D[0].size()-1)/2;
	for(int i = 0 ; i < ck.size();i++)
	{
		Z[ck[i]][ci[i]+ds] = cv[i];
	}

	for(int k =0 ; k < D.size(); k++)
	{
		for(int i =0 ; i < S.size(); i++)
		{
			double tmp = 0;
			for(int j=0 ; j< D[k].size(); j++)
			{
				tmp += D[k][j] * Z[k][i+j]; 
			}
			recon[i] += tmp;
		}
	}
	return recon;
}

void testFilePrecision(vector< vector<double> > in_D,vector<double> in_L, vector< vector<double> > in_S, vector< vector<short> > in_codek, vector< vector<short> > in_codei, vector< vector<double> > in_codeval)
{
	double total = 0;
	#pragma omp parallel for num_threads(T)
	for(int i = 0 ; i < in_S.size() ; i++)
	{
		vector<double> recon = genOldReconstruct(in_D,in_S[i],in_codek[i],in_codei[i],in_codeval[i]);
		float ls = genOldLoss(recon,in_S[i]);
		#pragma omp critical 
		total+=fabs(ls-in_L[i]);
	}
	printf("AvgLoss Lossy File Encode-Decode : %.10f\n",total/in_S.size());
}

double shrink(double beta, double lamb)
{
	if(beta>lamb) return beta - lamb;
	if(beta<-lamb) return beta + lamb;
	return 0;
}

double calcLoss()
{
	double loss=0;
	for(int ik = 0 ; ik<in_k; ik++)
	{	
		for(int i=0 ; i<ssize ;i++)
		{
			double x = s(ik,i);
			for(int k=0 ; k<K ;k++)
			{
				for(int j =-dsize; j<= dsize ;j++)
				{
					x-= d(k,ik,j) * z(k,ik,i+j);
				}
			}
			loss+= x*x;
		}
	}
	return loss;
}

void normalizeDictionary()
{
	for(int k=0; k < K;k++)
	{
		double sum = 0;
		for(int j=0; j<inD[2]; j++)
			for(int i=-dsize; i<=dsize; i++)
				sum+=d(k,j,i);
		for(int j=0; j<inD[2]; j++)
			for(int i=-dsize; i<=dsize; i++)
				d(k,j,i) /= sum;
	}
}

void initD()
{
	for(int k=0; k < K;k++)
		for(int j=0; j<inD[2]; j++)
			for(int i=-dsize; i<=dsize; i++)
				d(k,j,i) = rand()%1000;
	normalizeDictionary();
}

void initS(int i)
{
	vector<short> ck = in_codek[i];
	vector<short> ci = in_codei[i];
	vector<double> cv = in_codeval[i];

	for(int i=0 ; i<inD[2]; i++)
	{
		fill(S[i].begin(), S[i].end(), 0);
	}

	for(int i = 0 ; i < ck.size(); i++)
	{
		if(ci[i] >=in_sLen+in_dsize || ci[i] < -in_dsize)
			printf("Input Out of Range\n");
		s(ck[i],ci[i]+in_dsize) = cv[i];
	}
}

void normalizeS()
{

}

void initZ() 
{
	for(int k=0;k<K;k++)
		for(int ik=0;ik<in_k;ik++)
			for(int i=-dsize; i<inD[3]+dsize; i++)
				z(k,ik,i) = 0;
}

void initB()
{
	for(int k=0;k<K;k++)
	{
		for(int ik=0;ik<in_k;ik++)
		{
			for(int i = -dsize; i<ssize+dsize; i++)
			{
				b(k,ik,i) = 0;
				for(int j =-dsize ; j <= dsize ;j++)
				{
					if(i+j <0 || i+j >=ssize) continue;
					b(k,ik,i) += s(ik,i+j) * d(k,ik,-j);
				}
			}
		}
	}
}

int inference()
{
	int round = 0;
	int lastK=-1;
	int lastI=-1;
	double loss = calcLoss();
	printf("SLOSS=%f\n",loss);
	int negativeCount = 100;
	int positiveCount = 0;
	vector<int> lastDiff(100,0);


	vector< vector< vector<double> > > DIV  = vector< vector< vector<double> > >(K, vector<vector<double>>(in_k, vector<double>(ssize + dsize*2, 0 )));

	for(int k=0;k<K;k++)
	{
		for(int ik=0;ik<in_k;ik++)
		{
			for(int i= -dsize ; i< ssize + dsize; i++)
			{
				double sqdivider = 0;
				for(int j = -dsize; j<=dsize; j++)
				{
					if(i+j < 0 || i+j >= ssize) continue;
					sqdivider += (d(k,ik,-j) * d(k,ik,-j));
				}
				DIV[k][ik][i+dsize] = sqdivider;
			}
		}
	}

	while(true)
	{
		double maxV=-1;
		int maxK;
		int maxI;
		int maxIK;

		// calculate new Z //maximum diff Z
		for(int k=0;k<K;k++)
		{
			for(int ik=0;ik<in_k;ik++)
			{
				for(int i= -dsize ; i< ssize + dsize; i++)
				{
					zn(k,ik,i) = shrink(b(k,ik,i),LAMBDA) / DIV[k][ik][i+dsize];
					double fs = fabs(zn(k,ik,i) - z(k,ik,i) );
					if(fs > maxV )
					{
						if(lastK==k && lastI==i) printf("pick same \n");
						maxV=fs;
						maxK=k;
						maxIK=ik;
						maxI=i;
					}
				}
			}
		}

		// update beta
		// *** optimize to O(n^3) than O(n^4)
		double savedBeta = b(maxK,maxIK,maxI);
		for(int i=-2*dsize; i<= 2*dsize; i++)
		{
			int editing = i+maxI;
			if(editing < -dsize || editing >= ssize + dsize) continue;
			for(int k=0; k<K ;k++)
			{
				double total = 0;
				for(int r = -dsize + editing; r<=dsize + editing; r++)
				{
					if(r < 0 || r >= ssize) continue;
					for(int c = -dsize ; c<= dsize ;c++)
					{
						int zind = r+c;
						if( zind == maxI )
						{
							total += d(maxK,maxIK,c) * d(k,maxIK,-(r-editing));
						}
					}
				}
				b(k,maxIK,editing) -= (zn(maxK,maxIK,maxI)-z(maxK,maxIK,maxI)) * total;
			}
		}

		z(maxK,maxIK,maxI) = zn(maxK,maxIK,maxI);
		b(maxK,maxIK,maxI) = savedBeta;
		double newLoss = calcLoss();
		
		// stop conditions
		double diffLoss = newLoss-loss;
		if(diffLoss>0)
			printf("%d\tZLOSS=%.7f\tDiff=%.7f ****************\n",round,newLoss,diffLoss);	
		else
			printf("%d\tZLOSS=%.7f\tDiff=%.7f\n",round,newLoss,diffLoss);
		
		if( diffLoss > 0)
		{
			if(lastDiff[round%100] == 1) positiveCount--;
			else negativeCount--; 
			lastDiff[round%100] = 1;
			positiveCount++;
		}
		else
		{
			if(lastDiff[round%100] == 1) positiveCount--;
			else negativeCount--; 

			lastDiff[round%100] = -1;
			negativeCount++;
		}

		double percentBreak = (double)(positiveCount) / (positiveCount+negativeCount);
		if(round > inferenceMinRound )
		{
			if(percentBreak > inferencePercentBreak )
			{
				printf("Percent Break\n");
				break;
			}
			if(round  > inferenceMaxRound )
			{
				printf("MaxRound Break\n");
				break;
			}
			if(newLoss==0) break;
		}
		loss = newLoss;
		round++;
	}
	// cout <<"]\n";
	return round;
}

vector< vector<double> > reconstruct()
{
	vector< vector<double> > res(in_k, vector<double>(ssize,0));
	for(int ik=0; ik<in_k; ik++)
	{
		for(int i=0;i<ssize; i++)
		{
			double total = 0;
			for(int k=0;k<K;k++)
			{
				for(int d=-dsize; d<=dsize; d++)
				{
					total += z(k,ik,i+d) * d(k,ik,d);
				}
			}
			res[ik][i] = total;
		}
	}
	return res;
}

int learnDictionary()
{
	int round = 0 ;
	int special = 0;
	double dstepsize = dictstepsize;
	double lastLoss= calcLoss();
	int negativeCount = 100;
	int positiveCount = 0;
	vector<int> lastDiff(100,0);

	while(true)
	{
		vector< vector<double> > recon = reconstruct();
		vector< vector<double> > precalc(in_k,vector<double>(ssize,0));

		for(int ik=0;ik<in_k;ik++)
		{
			for(int i=0;i<ssize;i++)
			{
				precalc[ik][i] = s(ik,i) - recon[ik][i];
			}
		}
		
		for(int k=0;k<K;k++)
		{
			for(int ik=0;ik<in_k;ik++)
			{
				for(int d=-dsize;d<=dsize; d++)
				{
					nd(k,ik,d) = 0;
					for(int i = 0 ; i < ssize;i++)
					{
						nd(k,ik,d) += precalc[ik][i] * -z(k,ik,i+d);
					}
				}
			}
		}
		
		for(int k=0;k<K;k++)
		{
			for(int ik=0;ik<in_k;ik++)
			{
				for(int d=-dsize;d<=dsize; d++)
				{
					d(k,ik,d) -=  dstepsize * nd(k,ik,d);
				}
			}
		}
		normalizeDictionary();
		
		double loss= calcLoss();
		if(loss-lastLoss>0)
			printf("%d\tDLoss = %f %f = %f *******************\n",round,lastLoss,loss,loss-lastLoss);
		else
			printf("%d\tDLoss = %f %f = %f\n",round,lastLoss,loss,loss-lastLoss);

		if(loss-lastLoss > 0)
		{
			if(lastDiff[round%100] == 1) positiveCount--;
			else negativeCount--; 
			lastDiff[round%100] = 1;
			positiveCount++;
		}
		else
		{
			if(lastDiff[round%100] == 1) positiveCount--;
			else negativeCount--; 

			lastDiff[round%100] = -1;
			negativeCount++;
		}

		double percentBreak = (double)(positiveCount) / (positiveCount+negativeCount);
		if(round > dictMinRound)
		{
			if(fabs(loss-lastLoss) < dictDiffLossBreak)
			{
				printf("DiffLoss Break\n");
				break;
			}
			if(round > dictMaxRound)
			{
				printf("MaxRound Break\n");
				break;
			}
			if(percentBreak > dictPercentBreak)
			{
				printf("Percent Break\n");
				break;
			}
		}
		
		lastLoss=loss;
		round++;
	}
	return round;
}


int main(void)
{	
	int nSamples = parseBinary(&in_dsize,&in_k,&in_sLen,&in_D,&in_L,&in_S,&in_codek,&in_codei,&in_codeval);
	rd = shuffleArray(nSamples);

	int iD[]  = {nSamples,1,in_k,in_sLen+2*in_dsize};
	int oD[] = {nSamples,K,1,in_sLen+2*in_dsize+2*dsize};
	inD = iD;
	outD = oD;
	printf("input  : {%d, %d, %d, %d}\n",inD[0],inD[1],inD[2],inD[3]);
	printf("output : {%d, %d, %d, %d}\n",outD[0],outD[1],outD[2],outD[3]);
	printf("in_k : %d\n",in_k);
	printf("ssize : %d\n",ssize);
	printf("dsize : %d\n",dsize);
	printf("K : %d\n",K);
	// testFilePrecision(in_D,in_L,in_S,in_codek,in_codei,in_codeval);

	D  = vector< vector< vector<double> > >(K, vector<vector<double>>(in_k, vector<double>(dsize*2 +1, 0 )));
	Z  = vector< vector< vector<double> > >(K, vector<vector<double>>(in_k, vector<double>(outD[3], 0)));
	ZN = vector< vector< vector<double> > >(K, vector<vector<double>>(in_k, vector<double>(outD[3], 0)));
	B  = vector< vector< vector<double> > >(K, vector<vector<double>>(in_k, vector<double>(outD[3], 0)));
	S = vector< vector<double> >(inD[2], vector<double>(inD[3],0));
	ND  = vector< vector< vector<double> > >(K, vector<vector<double>>(in_k, vector<double>(dsize*2 +1, 0 )));

	initD();
	
	for(int i  =0 ; i < 3000; i++){
		initZ();
		initS(i);
		initB();
		inference();
		learnDictionary();
	}
	
	return 0;
}