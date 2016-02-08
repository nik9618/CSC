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


#define d(k,j,i) 	D[k][j][i+dsize]
#define s(j,i)		S[j][i]
#define z(j,i) 		Z[j][i+dsize]

#define T 30

using namespace std;

int * inD;
int * outD;

int in_dsize =0;
int in_k =0;
int in_sLen = 0;

int dsize = 10;
int ssize = 1214;
int K = 30;

vector< vector< vector<double> > >D;
vector< vector<double> > Z;
vector< vector<double> > B;
vector< vector<double> > S;
vector< vector< vector<double> > > ND;
vector< vector< vector<double> > > Zn;

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

void initD()
{
	for(int k=0; k < K;k++)
		for(int j=0; j<inD[2]; j++)
			for(int i=-dsize; i<=dsize; i++)
				d(k,j,i) = rand()%1000;
	
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
		s(ck[i],ci[i]+in_dsize) = cv[i];
	}
}

void initZ() 
{
	for(int k=0;k<K;k++)
		for(int i = -dsize;i<inD[3]+dsize;i++)
			z(k,i) = 0;
}

void initBeta()
{
	for(int k=0;k<K;k++)
	{
		for(int i = - dsize; i< ssize + dsize; i++)
		{
			b(k,i) = 0;
			for(int j =-dsize ; j <= dsize ;j++)
			{
				if(i+j <0 || i+j >=ssize) continue;
				for(int l=0; l<inD[2]; l++)
				{
					b(k,i) += s(l,i+j) * d(k,l,-j);
				}
			}
		}
	}
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
	
	// testFilePrecision(in_D,in_L,in_S,in_codek,in_codei,in_codeval);

	D = vector< vector< vector <double> > >(K, vector< vector <double> >(in_k, vector<double>(dsize*2 +1, 0 )));
	Z = vector< vector<double> >(K, vector<double>(outD[3], 0));
	B = vector< vector<double> >(K, vector<double>(outD[3], 0));
	S = vector< vector<double> >(inD[2], vector<double>(inD[3],0));

	initD();
	initZ();
	initS(0);
	initB();



	return 0;
}