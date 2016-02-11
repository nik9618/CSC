/*
REMEM : 
We have to admit this ... X is first sparse ... but R does not neccessary sparse after optimization ... but Z have to...

todo :
1. gen Z after avgLoss less than x. 
2. stack expanded dictionary 

done : 
0. calc average Loss
0. import multiples
0. debug z coding
1. optimize z coding.
2. multithreads.
3. print result.

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

#define binaryPath "/home/kanit/anomalydeep/dataout_lay2/data-%d.bin"
#define matlabfeedpath "/home/kanit/Dropbox/arrhythmia_project_shared/result_lay3/"
#define matlabdictpath "/home/kanit/Dropbox/arrhythmia_project_shared/result_lay3/1299.txt"
string datapath = "/home/kanit/anomalydeep/dataout_lay3/";

#define dm(j,k,i) 	DM[j][k][i+dsize]
#define d(t,j,k,i) 	D[t][j][k][i+dsize]
#define div(t,j,k,i) 	D[t][j][k][i+dsize]
#define s(t,j,i)		S[t][j][i]
#define z(t,k,i) 	Z[t][k][i+dsize]
#define zn(t,k,i) 	ZN[t][k][i+dsize]
#define b(t,k,i) 	B[t][k][i+dsize]
#define nd(t,j,k,i) 	ND[t][j][k][i+dsize]

#define T 2
#define LAMBDA 0.001
#define inferenceMinRound 50
#define inferencePercentBreak 0.2
#define inferenceMaxRound 500
#define inferenceDiffLossBreak 0.5

#define dictMinRound 10
#define dictMaxRound 20
#define dictDiffLossBreak 0.1
#define dictPercentBreak 0.1
double dictstepsize = 0.00001;

#define printLoss 0
#define willImportDict 0
#define genZmode 0

int in_dsize =0;	// input dsize
int in_k =0;		// input k
int in_prevk =0;	
int in_sLen = 0; 	// original signalLength

int dsize = 10;
int K = 100;

// -------
int ssize ;
vector< vector< vector<double> > >DM;
vector< vector< vector< vector<double> > > >D;
vector< vector< vector<double> > >Z;
vector< vector< vector<double> > >ZN;
vector< vector< vector<double> > >B;
vector< vector< vector<double> > >S;
vector< vector< vector< vector<double> > > >Zn;
vector< vector< vector< vector<double> > > >ND;
vector< vector< vector<double> > > recon;

vector< vector< vector<double> > > prev_D;
vector<double> prev_L;
vector< vector<short> > prev_codek;
vector< vector<short> > prev_codei;
vector< vector<double> > prev_codeval;
vector< vector<short> > in_codek;
vector< vector<short> > in_codei;
vector< vector<double> > in_codeval;

vector<double> tmp_L;
vector< vector<short> > tmp1_codek;
vector< vector<short> > tmp1_codei;
vector< vector<double> > tmp1_codeval;
vector< vector<short> > tmp2_codek;
vector< vector<short> > tmp2_codei;
vector< vector<double> > tmp2_codeval;

vector<int> rd;
ofstream * outfile;


vector<std::string> &split(const string &s, char delim, std::vector<std::string> &elems) {
	stringstream ss(s);
	string item;
	while (getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

vector<std::string> split(const string &s, char delim) {
	vector<string> elems;
	split(s, delim, elems);
	return elems;
}

int parseBinary(
	string infile,
	int *dsize,
	int *k,
	int *prevk,
	int *sLen,
	vector< vector< vector<double> > > *D,
	vector<double> *l,
	vector< vector<short> > *inck,
	vector< vector<short> > *inci,
	vector< vector<double> > *incv,
	vector< vector<short> > *outck,
	vector< vector<short> > *outci,
	vector< vector<double> > *outcv)
{
	ifstream file;
	file.open(infile,ifstream::binary);
	
	if (file.is_open()) {
	
		// parse Dict
		int recordLength = 0;
		int _dsize;
		int _k;
		int _ik;
		int _ssize;

		file.read((char*)&recordLength, 4);
		file.read((char*)&_k, 4);
		file.read((char*)&_ik, 4);
		file.read((char*)&_dsize, 4);
		file.read((char*)&_ssize, 4);
		
		*dsize = _dsize;
		*k = _k;
		*prevk = _ik;
		*sLen = _ssize+_dsize*2;

		// printf("%d %d %d %d",_dsize,_k,_ik, _ssize);
		vector< vector< vector<double> > > Dtmp = vector< vector< vector<double> >>(_k,vector< vector<double> >(_ik, vector<double>(_dsize*2+1,0)));

		for(int i = 0 ; i < _k ; i++)
		{
			for(int ik = 0 ; ik < _ik ; ik++)
			{
				for (int j = 0 ; j < _dsize*2+1;j++)
				{
					float d = 0;
					file.read((char*)&d,4);
					Dtmp[i][ik][j]=(double)d;
				}
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
		char * buf = (char*)malloc(10000000);
		
		int count = 0 ;
		while(file.good())
		{
			file.read((char*)&chunkLength, 4);
			if(!file.good()) break;
			file.read(buf, chunkLength);
			if(!file.good()) break;
			count++;
		}

		vector<double> ls = vector<double>(count,0);
		vector< vector<short> > in_codek = vector< vector<short> >(count, vector<short>(0,0) );
		vector< vector<short> > in_codei = vector< vector<short> >(count, vector<short>(0,0) );
		vector< vector<double> > in_codeval = vector< vector<double> >(count, vector<double>(0,0) );
		vector< vector<short> > out_codek = vector< vector<short> >(count, vector<short>(0,0) );
		vector< vector<short> > out_codei = vector< vector<short> >(count, vector<short>(0,0) );
		vector< vector<double> > out_codeval = vector< vector<double> >(count, vector<double>(0,0) );
		
		file.clear();
		file.seekg(0, ios::beg);
		
		//discard dictionary
		file.read((char*)&recordLength, 4);
		file.read(buf, recordLength);

		
		for(int i =0 ;  i < count ;i++)
		{
			file.read((char*)&chunkLength, 4);
			file.read(buf, chunkLength);	
			
			int idx = 0 ;
			int take = 0;

			// loss 
			take=8; memcpy(&loss,buf+idx,take); idx+=take;
			ls[i] = loss;

			// input signal len
			take=2; memcpy(&sigLen,buf+idx,take); idx+=take;
			
			// input signal array
			in_codek[i].resize(sigLen);
			in_codei[i].resize(sigLen);
			in_codeval[i].resize(sigLen);
			for(int j=0 ; j<sigLen; j++)
			{	
				take=2; memcpy(&stmp,buf+idx,take); idx+=take; 
				in_codek[i][j]=stmp;
				take=2; memcpy(&stmp,buf+idx,take); idx+=take; 
				in_codei[i][j]=stmp;
				take=4; memcpy(&ftmp,buf+idx,take); idx+=take;
				in_codeval[i][j]=ftmp;
			}

			// output signal len
			take=2; memcpy(&zLen,buf+idx,take); idx+=take;


			// output signal array
			out_codek[i].resize(zLen);
			out_codei[i].resize(zLen);
			out_codeval[i].resize(zLen);
			for(int j=0 ; j<zLen; j++)
			{	
				take=2; memcpy(&stmp,buf+idx,take); idx+=take; 
				out_codek[i][j]=stmp;
				take=2; memcpy(&stmp,buf+idx,take); idx+=take; 
				out_codei[i][j]=stmp;
				take=4; memcpy(&ftmp,buf+idx,take); idx+=take;
				out_codeval[i][j]=ftmp;
			}
		}

		*l = ls;
		*inck = in_codek;
		*inci = in_codei;
		*incv = in_codeval;
		*outck = out_codek;
		*outci = out_codei;
		*outcv = out_codeval;
		
		return count;
	}
	else
	{
		return 0;
	}
}

void testFilePrecision(vector< vector< vector<double> > > in_D,vector<double> in_L, vector< vector<short> > prev_codek, vector< vector<short> > prev_codei, vector< vector<double> > prev_codeval, vector< vector<short> > in_codek, vector< vector<short> > in_codei, vector< vector<double> > in_codeval)
{	
	double sumLoss = 0;
	#pragma omp parallel for num_threads(T)
	for(int i = 0 ; i < prev_codek.size() ; i++)
	{
		//gen X Z 
		vector< vector<double> > X(in_prevk, vector<double>(ssize-2*in_dsize,0) );
		vector< vector<double> > R(in_prevk, vector<double>(ssize-2*in_dsize,0) );
		vector< vector<double> > Z(in_k, vector<double>(ssize,0) );

		//uncompress X Z
		for (int j=0; j<prev_codei[i].size(); j++)
		{
			X[prev_codek[i][j]][prev_codei[i][j]] = prev_codeval[i][j];
 		}

 		for (int j=0; j<in_codei[i].size(); j++)
		{
			Z[in_codek[i][j]][in_codei[i][j]+in_dsize] = in_codeval[i][j];
 		}
		
		//reconstruct 
		double loss = 0;
		for(int j=0;j<ssize-2*in_dsize; j++)
		{
			for(int pk=0; pk<in_prevk; pk++)
			{
				double total = 0;
				for(int ik=0; ik<in_k; ik++)
				{
					for(int d=-in_dsize; d<=in_dsize; d++)
					{
						total += Z[ik][j+d+in_dsize] * in_D[ik][pk][d+in_dsize];
					}
				}
				loss += (X[pk][j] - total) * (X[pk][j] - total);
			}
		}
		if(fabs(in_L[i]-loss )> 0.001)printf("FAIL RECONCILE : %f %f %f\n",in_L[i]-loss ,loss, in_L[i]); 
		#pragma omp critical
		{
			sumLoss += in_L[i]-loss;
		}
	}
	printf("AvgLoss Lossy File Encode-Decode : %.10f\n",sumLoss/ prev_codek.size());
}

double shrink(double beta, double lamb)
{
	if(beta>lamb) return beta - lamb;
	if(beta<-lamb) return beta + lamb;
	return 0;
}

vector< vector<double> > reconstruct(int t)
{
	double total;	
	for(int i=0;i<ssize; i++)
	{
		for(int ik=0; ik<in_k; ik++)
		{
			total=0;
			for(int k=0;k<K;k++)
			{
				for(int d=-dsize; d<=dsize; d++)
				{
					total += z(t,k,i+d) * d(t,k,ik,d);
				}
			}
			recon[t][ik][i] = total;
		}
	}
	return recon[t];
}

double calcLoss(int t)
{
	double loss=0;

	for(int i=0;i<ssize; i++)
	{
		for(int ik=0; ik<in_k; ik++)
		{
			double total = s(t,ik,i);
			for(int k=0;k<K;k++)
			{
				for(int d=-dsize; d<=dsize; d++)
				{
					total -= z(t,k,i+d) * d(t,k,ik,d);
				}
			}
			loss += total*total;
		}
	}
	return loss;
}

void normalizeDictionary(int t)
{
	for(int k=0; k < K;k++)
	{
		double sum = 0;
		for(int j=0; j<in_k; j++)
			for(int i=-dsize; i<=dsize; i++)
				sum+=d(t,k,j,i);
		for(int j=0; j<in_k; j++)
			for(int i=-dsize; i<=dsize; i++)
				d(t,k,j,i) /= sum;
	}
}

void syncDictionary(int t)
{
	for(int k = 0 ; k < K ; k++)
		for(int ik = 0 ; ik < in_k ; ik++)
			for(int i = -dsize; i <=dsize ; i++ )
				dm(k,ik,i) = 0.9*dm(k,ik,i) + 0.1*d(t,k,ik,i);
	
	for(int k = 0 ; k < K ; k++)
		for(int ik = 0 ; ik < in_k ; ik++)
			for(int i = -dsize; i <=dsize ; i++ )
				d(t,k,ik,i)=dm(k,ik,i);

	normalizeDictionary(t);
}

void importDictionary()
{
	ifstream infile(matlabdictpath);	
	string line;
	
	//skip 9 lines
	for(int i = 0 ; i < 9 ;i++) getline(infile, line);

	int k=0;
	while(getline(infile, line))
	{
		int st = line.find("[");
		int ed = line.find("]");
		string nums = line.substr(st,ed-st);
		vector<string> x = split(nums, ' ');
		int i = 0;
		for(int ik=0;ik<in_k;ik++)
		{
			for(int d=-dsize; d<=dsize;d++)
			{
				if(x[i][0] == ';')
					x[i] = x[i].substr(1,strlen(x[i].c_str())-1);
				dm(k,ik,d) = strtod(x[i].c_str(), NULL);
				i++;
			}
		}
		k++;
	}
	printf("Imported D : %d\n",K);
	for(int k=0; k < K;k++)
	{
		for(int j=0; j<in_k; j++)
		{
			for(int i=-dsize; i<=dsize; i++)
			{
				for(int t=0; t<T; t++)
				{
					d(t,k,j,i) = dm(k,j,i);	
				}
			}
		}
	}
}

void initD()
{
	for(int k=0; k < K;k++)
	{
		for(int j=0; j<in_k; j++)
		{
			for(int i=-dsize; i<=dsize; i++)
			{
				dm(k,j,i) = rand()%1000;
			}
		}
	}

	for(int k=0; k < K;k++)
	{
		double sum = 0;
		for(int j=0; j<in_k; j++)
			for(int i=-dsize; i<=dsize; i++)
				sum+=dm(k,j,i);
		for(int j=0; j<in_k; j++)
			for(int i=-dsize; i<=dsize; i++)
				dm(k,j,i) /= sum;
	}

	for(int k=0; k < K;k++)
	{
		for(int j=0; j<in_k; j++)
		{
			for(int i=-dsize; i<=dsize; i++)
			{
				for(int t=0; t<T; t++)
				{
					d(t,k,j,i) = dm(k,j,i);	
				}
			}
		}
	}
}

void initS(int t, int i)
{
	vector<short> ck = in_codek[i];
	vector<short> ci = in_codei[i];
	vector<double> cv = in_codeval[i];

	for(int i=0 ; i<in_k; i++)
	{
		fill(S[t][i].begin(), S[t][i].end(), 0);
	}

	for(int i = 0 ; i < ck.size(); i++)
	{
		if(ci[i] >=in_sLen+in_dsize || ci[i] < -in_dsize)
			printf("Input Out of Range\n");
		s(t,ck[i],ci[i]+in_dsize) = cv[i];
	}
}

void normalizeS(int t)
{
	double sum = 0;
	double sum2 = 0;
	for(int ik = 0 ; ik < in_k;ik++)
	{
		for(int i = 0 ; i < ssize*2;i++)
		{
			sum+=s(t,ik,i);
			sum2 =(s(t,ik,i)*s(t,ik,i))+sum2;
		} 
	}
	double mean = sum/ssize;
	double std = sqrt(sum2/ssize-mean*mean);
	
	if(std == std && std > 0.0001)
	{
		for(int ik = 0 ; ik < in_k;ik++)
		{
			for(int i = 0 ; i < ssize*2;i++)
			{
				s(t,ik,i) = s(t,ik,i)/std;
				// s(t,i) = ((s(t,i)-mean));
				// if(s(t,i)!=s(t,i)) anynan++;
			}
		}
	}

	for(int ik = 0 ; ik < in_k;ik++)
	{
		for(int i = 0 ; i < ssize;i++)
		{
			s(t,ik,i) = abs(s(t,ik,2*i))>abs(s(t,ik,2*i+1)) ? s(t,ik,2*i) : s(t,ik,2*i+1);
		}
	}
}

void initZ(int t) 
{
	for(int k=0;k<K;k++)
		for(int i=-dsize; i<ssize+dsize; i++)
			z(t,k,i) = 0;
}

void initB(int t)
{
	for(int k=0;k<K;k++)
	{
		for(int i = -dsize; i<ssize+dsize; i++)
		{
			b(t,k,i) = 0;
			for(int ik=0;ik<in_k;ik++)
			{
				for(int j =-dsize ; j <= dsize ;j++)
				{
					if(i+j <0 || i+j >=ssize) continue;
					b(t,k,i) += s(t,ik,i+j) * d(t,k,ik,-j);
				}
			}
		}
	}
}

vector<double> inference(int t,double initLoss)
{
	int round = 0;
	int lastK=-1;
	int lastI=-1;
	double loss = initLoss;

	vector< vector<double> > DIV(K, vector<double>(ssize + dsize*2, 0 ));

	for(int k=0;k<K;k++)
	{
	
		for(int i= -dsize ; i< ssize + dsize; i++)
		{
			double sqdivider = 0;
			for(int ik=0;ik<in_k;ik++)
			{
				for(int j = -dsize; j<=dsize; j++)
				{
					if(i+j < 0 || i+j >= ssize) continue;
					sqdivider += (d(t,k,ik,-j) * d(t,k,ik,-j));
				}
			}
			DIV[k][i+dsize] = 1/sqdivider;
		}
	}

	while(true)
	{
		double maxV=-1;
		int maxK;
		int maxI;

		// calculate new Z //maximum diff Z
		for(int k=0;k<K;k++)
		{
			for(int i= -dsize ; i< ssize + dsize; i++)
			{
				zn(t,k,i) = shrink(b(t,k,i),LAMBDA) * DIV[k][i+dsize];
				double fs = fabs(zn(t,k,i) - z(t,k,i) );
				if(fs > maxV )
				{
					// if(lastK==k && lastI==i) printf("pick same \n");
					maxV=fs;
					maxK=k;
					maxI=i;
				}
			}
		}

		// update beta
		double savedBeta = b(t,maxK,maxI);
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
							for(int ik=0; ik<in_k; ik++)
							{
								total += d(t,maxK,ik,c) * d(t,k,ik,-(r-editing));
							}
						}
					}
				}
				b(t,k,editing) -= (zn(t,maxK,maxI)-z(t,maxK,maxI)) * total;
			}
		}

		z(t,maxK,maxI) = zn(t,maxK,maxI);
		b(t,maxK,maxI) = savedBeta;
		
		if(round %40==39)
		{
			double newLoss = calcLoss(t);
			
			// stop conditions
			double diffLoss = newLoss-loss;
			if(t==0 && printLoss)
			{
				if(diffLoss>0)
					printf("%d %d\tZLOSS=%.7f\tDiff=%.7f ****************\n",t,round,newLoss,diffLoss);	
				else
					printf("%d %d\tZLOSS=%.7f\tDiff=%.7f\n",t,round,newLoss,diffLoss);
			}

			if(round > inferenceMinRound )
			{
				if( fabs(diffLoss) < inferenceDiffLossBreak )
				{
					// printf("Percent Break\n");
					break;
				}
				if(round  > inferenceMaxRound )
				{
					// printf("MaxRound Break\n");
					break;
				}
				if(newLoss==0) break;
			}

			loss = newLoss;
		}
		round++;
	}
	// cout <<"]\n";
	vector<double> ans(2);
	ans[0]=round;
	ans[1]=loss;
	return ans;
}

vector<double> learnDictionary(int t, double initLoss)
{
	int round = 0 ;
	int special = 0;
	double dstepsize = dictstepsize;
	double lastLoss= initLoss;
	int negativeCount = 100;
	int positiveCount = 0;
	vector<int> lastDiff(100,0);

	// vector< vector<double> > recon = reconstruct(t);
	vector< vector<double> > precalc(in_k,vector<double>(ssize,0));

	while(true)
	{
		//tied in to boost performance
		double loss=0;
		double tmpRecon;
		for(int i=0;i<ssize; i++)
		{
			for(int ik=0; ik<in_k; ik++)
			{
				tmpRecon = 0;
				for(int k=0;k<K;k++)
				{
					for(int d=-dsize; d<=dsize; d++)
					{
						tmpRecon +=  z(t,k,i+d) * d(t,k,ik,d);
					}
				}
				double diff = s(t,ik,i)-tmpRecon;
				precalc[ik][i] = diff;
				loss += (diff)*(diff);
				recon[t][ik][i]=tmpRecon;
			}
		}
		
		for(int k=0;k<K;k++)
		{
			for(int ik=0;ik<in_k;ik++)
			{
				for(int d=-dsize;d<=dsize; d++)
				{
					nd(t,k,ik,d) = 0;
					for(int i = 0 ; i < ssize;i++)
					{
						nd(t,k,ik,d) += precalc[ik][i] * -z(t,k,i+d);
					}
					d(t,k,ik,d) -=  dstepsize * nd(t,k,ik,d);
				}
			}
		}
		
		syncDictionary(t);
		
		if(t==0 && printLoss)
		{
			if(loss-lastLoss>0)
				printf("%d\tDLoss = %f %f = %f *******************\n",round,lastLoss,loss,loss-lastLoss);
			else
				printf("%d\tDLoss = %f %f = %f\n",round,lastLoss,loss,loss-lastLoss);
		}
		
		if(loss-lastLoss > 0)
		{
			dstepsize/=2;
		}

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
				// printf("DiffLoss Break\n");
				break;
			}
			if(round > dictMaxRound)
			{
				// printf("MaxRound Break\n");
				break;
			}
			if(percentBreak > dictPercentBreak)
			{
				// printf("Percent Break\n");
				break;
			}
		}
		
		lastLoss=loss;
		round++;
	}
	vector<double> ans(2);
	ans[0]=round;
	ans[1]=lastLoss;
	return ans;
}

void reportTesting(int t,double loss,double zround, double dround, int fileID)
{
	ofstream myfile;
	
	// std::stringstream ss;
	// ss <<"mkdir -p /home/kanit/anomalydeep/result_lay2";
	// std::string genfolder = ss.str();
	// int x = system(genfolder.c_str());

	myfile.open (matlabfeedpath + to_string(fileID) +".txt");
	
	myfile << "Loss="<<loss<<"\n";
	myfile << "Zround="<<zround<<"\n";
	myfile << "Dround="<<dround<<"\n";
	int morezero =0;
	int all = 0;
	for(int k =0 ; k < K ; k++)
	{
		for(int i = -dsize ; i < ssize+dsize;i++)
		{
			if(z(t,k,i)!=0)
			{
				morezero++;
			}
			all++;
		}
	}
	myfile << "sparsity="<<morezero<<"/"<<all<<"\n\n";
	myfile << "S" << "=[";	
	for(int ik = 0 ; ik < in_k ; ik++)
	{
		for(int i = 0 ; i < ssize ; i++)
		{
			myfile << s(t,ik,i)<<" ";
		}
		myfile << ";";
	}
	myfile << "]\n\n";

	vector<vector<double> > rec = reconstruct(t);
	myfile << "R" << "=[";	
	for(int ik = 0 ; ik < in_k ; ik++)
	{
		for(int i = 0 ; i < ssize ; i++)
		{
			myfile << rec[ik][i]<<" ";
		}
		myfile << ";";
	}
	myfile << "]\n\n";
	
	for(int i = 0 ; i < K ; i++)
	{
		myfile << "D" <<i<< "=["<<setprecision(8) ;
		for(int ik = 0 ; ik < in_k ; ik++)
		{
			for (int j = -dsize; j<=dsize ;j++)
			{
				myfile << d(t,i,ik,j)<<" ";
			}
			myfile << ";";
		}
		myfile << "]\n";
	}
	
	myfile.close();
}

void mergeInput( vector<double> sl, vector< vector<short> > pck, vector< vector<short> > pci, vector< vector<double> > pcv,vector< vector<short> > ick, vector< vector<short> > ici, vector< vector<double> > icv)
{
	for(int i = 0;  i< ick.size() ;i++)
	{
		in_codek.push_back(ick[i]);
		in_codei.push_back(ici[i]);
		in_codeval.push_back(icv[i]);
		
		prev_L.push_back(sl[i]);

		prev_codek.push_back(pck[i]);
		prev_codei.push_back(pci[i]);
		prev_codeval.push_back(pcv[i]);
	}
}

void writeHead(int t)
{
	int recordLen = K*in_k*(dsize*2+1)*4 +4+4+4+4;
	outfile[t].write((char*) &recordLen, 4);
	outfile[t].write((char*) &K, 4);
	outfile[t].write((char*) &in_k, 4);
	outfile[t].write((char*) &dsize, 4);
	outfile[t].write((char*) &ssize, 4);
	
	for(int k =0 ; k < K ; k++)
	{
		for(int ik =0 ; ik < in_k ; ik++)
		{
			for(int i=-dsize ; i <= dsize ; i++)
			{
				float dd = dm(k,ik,i);
				outfile[t].write((char*) &dd,4);
			}
		}
	}
	outfile[t].flush();
}

void writeFile(int t,double loss)
{
	int count = 0;
	
	vector<float> inval(0,0);
	vector<short> inkindex(0,0);
	vector<short> iniindex(0,0);
	
	vector<float> outval(0,0);
	vector<short> outkindex(0,0);
	vector<short> outiindex(0,0);
	
	for(int ik=0 ; ik < in_k ; ik++)
	{
		for(int i=0;i<ssize;i++)
		{
			if(abs(s(t,ik,i))>0.000001)
			{
				inval.push_back((float)s(t,ik,i));
				inkindex.push_back(ik);
				iniindex.push_back(i);
			}
		}
	}

	for(int k=0 ; k<K; k++)
	{
		for(int i=-dsize; i<ssize+dsize; i++)
		{
			if(abs(z(t,k,i))>0.000001)
			{
				outval.push_back((float)z(t,k,i));
				outkindex.push_back(k);
				outiindex.push_back(i);
			}
		}
	}

	int recordLen = 8 + (2 + inval.size()*(2+2+4)) + (2 + (2+2+4) * outval.size());
	
	outfile[t].write((char*) &recordLen, 4);
	outfile[t].write((char*) &loss, 8);
	
	//z input of the layer
	short slen_short = (short)inval.size();	
	outfile[t].write((char*) &slen_short, 2);

	for(short i=0 ; i < slen_short; i++)
	{
		outfile[t].write((char*) &inkindex[i], 2);
		outfile[t].write((char*) &iniindex[i], 2);
		outfile[t].write((char*) &inval[i], 4);
	}
	
	//z output of the layer
	short zlen_short = (short)outval.size();	
	outfile[t].write((char*) &zlen_short, 2);

	for(short i=0 ; i < zlen_short; i++)
	{
		outfile[t].write((char*) &outkindex[i], 2);
		outfile[t].write((char*) &outiindex[i], 2);
		outfile[t].write((char*) &outval[i], 4);
	}
	outfile[t].flush();
}

void initGenZFile()
{
	outfile = new ofstream[T];
	for(int i = 0; i< T ; i++)
	{
		char outname[150];
		sprintf(outname,"%sdata-%d.bin",datapath.c_str(),i);
		outfile[i].open(outname,std::ofstream::binary);
		writeHead(i);
	}
}

int main(void)
{	
	int nSamples=0;
	
	//main data
	char dataName[100];
	sprintf(dataName,binaryPath,0);
	nSamples += parseBinary(
		dataName,
		&in_dsize,
		&in_k,
		&in_prevk,
		&in_sLen,
		&prev_D,
		&prev_L,
		&prev_codek,
		&prev_codei,
		&prev_codeval,
		&in_codek,
		&in_codei,
		&in_codeval);
	
	ssize = (int)in_sLen;


	//more data
	for(int i =1 ; i<30;i++)
	{
		sprintf(dataName,binaryPath,i);
		nSamples += parseBinary(
			dataName,
			&in_dsize,
			&in_k,
			&in_prevk,
			&in_sLen,
			&prev_D,
			&tmp_L,
			&tmp1_codek,
			&tmp1_codei,
			&tmp1_codeval,
			&tmp2_codek,
			&tmp2_codei,
			&tmp2_codeval);
		mergeInput(tmp_L, tmp1_codek, tmp1_codei, tmp1_codeval, tmp2_codek, tmp2_codei, tmp2_codeval);
	}

	printf("samples : %d\n",nSamples);
	printf("prev_k : %d\n",in_prevk);
	printf("in_k : %d\n",in_k);
	printf("ssize : %d\n",ssize);
	printf("dsize : %d\n",dsize);
	printf("OldDsize : %d\n",in_dsize);
	printf("K : %d\n",K);
	printf("T : %d\n",T);

	// testFilePrecision(prev_D,prev_L,prev_codek,prev_codei,prev_codeval,in_codek,in_codei,in_codeval);
	// return 0;

	DM = vector< vector< vector<double> > >(K, vector<vector<double>>(in_k, vector<double>(dsize*2 +1, 0 )));
	D  = vector< vector< vector< vector<double> > > >(T,vector< vector< vector<double> > >(K, vector<vector<double>>(in_k, vector<double>(dsize*2 +1, 0 ))));
	Z  = vector< vector< vector<double> > >(T,vector< vector<double > >(K, vector<double>(ssize+2*dsize, 0)));
	ZN = vector< vector< vector<double> > >(T,vector< vector<double > >(K, vector<double>(ssize+2*dsize, 0)));
	B  = vector< vector< vector<double> > >(T,vector< vector<double > >(K, vector<double>(ssize+2*dsize, 0)));
	S  = vector< vector< vector<double> > >(T, vector< vector<double> >(in_k, vector<double>(ssize,0)));
	ND  = vector< vector< vector< vector<double> > > >(T, vector< vector< vector<double> > >(K, vector<vector<double>>(in_k, vector<double>(dsize*2 +1, 0 ))));
	recon = vector< vector< vector<double> > >(T,vector< vector<double> >(in_k, vector<double>(ssize,0)));
	
	ssize=310; // max pooling....
	printf("ssize : %d\n",ssize);
	//parameters settled

	if(willImportDict)
		importDictionary();
	else
		initD();
	
	int round = 0;
	int round0 = 0;
	
	double sum100= 0;
	vector<double> data100(100,0);
	double num100=0;
	double avgLoss;

	if(genZmode)
	{
		initGenZFile();
	}

	#pragma omp parallel for num_threads(T)
	for(int i  =0 ; i < nSamples; i++){
		
		int t = omp_get_thread_num();
		initS(t,i);
		normalizeS(t);
		initZ(t);
		initB(t);
		
		double l1 = calcLoss(t);
		if(t==0 && avgLoss == -1) avgLoss = l1;

		vector<double> a1 = inference(t,l1);
		
		if(genZmode==0)
		{
			vector<double> a2 = learnDictionary(t,a1[1]);
			
			#pragma omp critical
			{
				//average Loss
				sum100 -= data100[round%100];
				data100[round%100] =a2[1];
				sum100 += a2[1];
				num100++;
				if(num100>100)num100=100;
				avgLoss = sum100/num100;
				
				round++;
				printf("%d\t1> %d\t2> %f\t%.0f\t3> %f\t%.0f\t4> %f \t5> %f\n",t,round,l1,a1[0],a1[1],a2[0],a2[1],avgLoss);
			}
			
			if(t==0)
			{
				if(round0%5==0)
					reportTesting(t,a2[1],a1[0],a2[0], round);
				round0++;
			}
		}
		else
		{
			double immediateLoss = calcLoss(t);
			writeFile(t,immediateLoss);
			printf("%d\t1> %d\t2> %f\t3> %.0f\t4> %f\n",t,round,l1,a1[0],immediateLoss);
			#pragma omp critical
			round++;
		}
	}
	
	return 0;
}