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

#define margin 		100 // becareful margin must me at least 3*dsize
#define os(t,i) 	OS[t][i+margin]
#define s(t,i) 		S[t][i+margin]
#define d(t,k,i) 	D[t][k][i+dsize]
#define dm(k,i) 	DMASTER[k][i+dsize]
#define nd(t,k,i) 	ND[t][k][i+dsize]
#define z(t,k,i) 	Z[t][k][i+margin]
#define zn(t,k,i) 	Zn[t][k][i+margin]
#define b(t,k,i) 	B[t][k][i+margin]

#define flagImportDict true
#define maxData 2000000
#define sigSize 1250
#define bandFilterSize 47 //@ 2 hz
#define MAFilterSize 5 
#define MAFilter 0.20 

//TO TUNE
// lambda dictsize num_dict
double LAMBDA = 0.05;
int dsize = 4;
int T = 30;
int K = 40;

int ssize = sigSize - bandFilterSize - MAFilterSize + 2;
double dictstepsize = 0.00001;
int dictRound = 5;
int inferenceMinRound = 100;
int maxCoordinateDescentLoop = 2000;
// double diffLossSPC = 0.0002;

int miniBatch = 5;

double filter[47] = {-8.15926288e-04,  -9.76750628e-04,  -1.26060467e-03,
        -1.69918730e-03,  -2.32002576e-03,  -3.14491716e-03,
        -4.18856455e-03,  -5.45747532e-03,  -6.94917936e-03,
        -8.65181110e-03,  -1.05440838e-02,  -1.25956675e-02,
        -1.47679632e-02,  -1.70152499e-02,  -1.92861613e-02,
        -2.15254358e-02,  -2.36758695e-02,  -2.56803928e-02,
        -2.74841826e-02,  -2.90367233e-02,  -3.02937271e-02,
        -3.12188334e-02,  -3.17850122e-02,   9.67262288e-01,
        -3.17850122e-02,  -3.12188334e-02,  -3.02937271e-02,
        -2.90367233e-02,  -2.74841826e-02,  -2.56803928e-02,
        -2.36758695e-02,  -2.15254358e-02,  -1.92861613e-02,
        -1.70152499e-02,  -1.47679632e-02,  -1.25956675e-02,
        -1.05440838e-02,  -8.65181110e-03,  -6.94917936e-03,
        -5.45747532e-03,  -4.18856455e-03,  -3.14491716e-03,
        -2.32002576e-03,  -1.69918730e-03,  -1.26060467e-03,
        -9.76750628e-04,  -8.15926288e-04};

using namespace std;

string resultpath = "/home/kanit/anomalydeep/";

string datapath = "/home/kanit/anomalydeep/dataout/";

vector< vector<double> > S;
vector< vector<double> > OS;
vector< vector< vector<double> > > D;
vector< vector<double> > DMASTER;
vector< vector< vector<double> > > Z;
vector< vector< vector<double> > > B;

vector< vector<double> > Zct;
vector< vector< vector<double> > > ND;
vector< vector< vector<double> > > Zn;

ofstream * outfile;

double shrink(double beta, double lamb)
{
	if(beta>lamb) return beta - lamb;
	if(beta<-lamb) return beta + lamb;
	return 0;
}

double calcLoss(int t)
{
	double loss=0;
	for(int i =0 ; i < ssize ;i++)
	{
		double x = s(t,i);
		for(int k = 0 ; k <  K ; k++)
		{
			for(int j =-dsize; j<= dsize ;j++)
			{
				x-= d(t,k,j) * z(t,k,i+j);
			}
		}
		loss+= x*x;
	}
	return loss;
}


void reportZ(int t)
{
	for(int k = 0 ; k < K ; k++)
	{
		cout << "Z"<<k<<":\t";
		for(int i = -dsize; i<= ssize+dsize ;i++)
		{
			cout << z(t,k,i) <<"\t";
		}
		cout<<endl;
	}
}

void reportD(int t)
{
	for(int k = 0 ; k < K ; k++)
	{
		for(int i = - dsize; i<= dsize ;i++)
		{
			cout << d(t,k,i) <<"\t";
		}
		cout<<endl;
	}
}

void reportB(int t)
{
	for(int k=0;k<K;k++)
	{
		for(int i = - dsize; i< ssize + dsize; i++)
		{
			cout << b(t,k,i)<<"\t";
		}
		cout<<endl;
	}
}

void reportS(int t) 
{
	for(int i = 0; i < ssize;i++) cout << s(t,i) <<" ";
	cout<<endl<<endl;
}

vector<double> reconstruct(int t)
{
	vector<double> res(ssize,0);
	for(int i =0 ; i < ssize ;i++)
	{
		double x = 0;
		for(int k = 0 ; k <  K ; k++)
		{
			for(int j =-dsize; j<= dsize ;j++)
			{
				x += d(t,k,j) * z(t,k,i+j);
			}
		}
		res[i] = x;
	}
	return res;
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

vector< vector<double> > timeSeries(string path)
{
	ifstream myReadFile;
	
	myReadFile.open(path.c_str(),std::ifstream::binary);
	string line;
	if (myReadFile.is_open()) {
		
		int countII=0;
		int countV=0;
		int countABP=0;

		char sigName[32];
		long sigStart = 0;
		short freq = 0;
		short sigLen = 0;
		char hasII = 0;
		char hasV = 0;
		char hasABP = 0;
		char unuse = 0;
		double baseII;
		double gainII;
		double baseV;
		double gainV;
		double baseABP;
		double gainABP;

		int recordLength = 0;

		short shII[1250];
		short shV[1250];
		short shABP[1250];

		//fixed for fast !
		countII = 7260032;

		myReadFile.seekg(0, ios::beg);
		vector< vector<double> > v(maxData,vector<double>(1250,0));

		int i = 0;
		int placeIndex =0;
		int takeEvery = countII / maxData;
		while (!myReadFile.eof())
		{
			myReadFile.read((char*)&recordLength, 4);

			myReadFile.read((char*)&sigName, 32);
			myReadFile.read((char*)&sigStart, 8);
			myReadFile.read((char*)&freq, 2);
			myReadFile.read((char*)&sigLen, 2);
			
			myReadFile.read((char*)&hasII, 1);
			myReadFile.read((char*)&hasV, 1);
			myReadFile.read((char*)&hasABP, 1);
			myReadFile.read((char*)&unuse, 1);
			
			myReadFile.read((char*)&baseII, 8);
			myReadFile.read((char*)&gainII, 8);
			myReadFile.read((char*)&baseV, 8);
			myReadFile.read((char*)&gainV, 8);
			myReadFile.read((char*)&baseABP, 8);
			myReadFile.read((char*)&gainABP, 8);

			if(hasII==1) myReadFile.read((char*)&shII, 1250*2);
			if(hasV==1) myReadFile.read((char*)&shV, 1250*2);
			if(hasABP==1) myReadFile.read((char*)&shABP, 1250*2);

			if(hasII==1)
			{
				if(i % takeEvery == 0 && placeIndex < maxData)
				{
					for(int j=0; j< 1250;j++)
					{
						v[placeIndex][j] = (shII[j] - baseII) * gainII;
					}
					placeIndex++;
				}
				i++;
			}
			// break;
			if(i==countII) break;
		}
		printf("read file done(%d)\n",maxData);
		return v;
	}
	else
	{
		cout << "404 file not found ! :(" << endl;
		vector< vector<double> > v(0,vector<double>(0,0));
		return v;
	}
}

void reportTesting(int t, string folder, string name,int infround)
{
	ofstream myfile;
	
	std::stringstream ss;
	ss <<"mkdir -p "<< resultpath << folder;
	std::string genfolder = ss.str();
	int x = system(genfolder.c_str());
	myfile.open (resultpath + folder + "/" + name +".txt");
	
	// myfile << "OS" << "=[";	
	// for(int i = 0 ; i < ssize ; i++)
	// {
	// 	myfile << os(t,i)<<" ";
	// }
	// myfile << "]\n";
	myfile << "loss="<<calcLoss(t)<<"\n";
	myfile << "inferround="<<infround<<"\n";
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
	for(int i = 0 ; i < ssize ; i++)
	{
		myfile << s(t,i)<<" ";
	}
	myfile << "]\n\n";

	vector<double> rec = reconstruct(t);
	myfile << "R" << "=[";	
	for(int i = 0 ; i < ssize ; i++)
	{
		myfile << rec[i]<<" ";
	}
	myfile << "]\n\n";
	
	// for(int k=0;k<K;k++)
	// {
	// 	myfile << "ND" << k << "=["<<setprecision(3);	
	// 	for(int i=-dsize;i<=dsize; i++)
	// 	{
	// 		myfile << nd(t,k,i) <<" ";
	// 	}
	// 	myfile << "]\n";
	// 	break;
	// }
	// for(int k=0;k<K;k++)
	// {
	// 	myfile << "Z" << k << "=["<<setprecision(3);	
	// 	for(int i=-dsize;i<ssize+dsize; i++)
	// 	{
	// 		myfile << z(t,k,i) <<" ";
	// 	}
	// 	myfile << "]\n";
	// 	break;
	// }

	myfile << "D" << "=["<<setprecision(10) ;
	for(int i = 0 ; i < K ; i++)
	{
		for (int j = -dsize; j<=dsize ;j++)
		{
			myfile << d(t,i,j)<<" ";
		}
		myfile << ";";
	}
	myfile << "]\n";
	myfile.close();
}

void initD()
{
	for(int k=0; k < K;k++)
		for(int i=-dsize; i<=dsize; i++)
			dm(k,i) = rand()%1000;
	for(int k=0; k < K;k++)
	{
		double sum = 0;
		for(int i=-dsize; i<=dsize; i++)
			sum+=dm(k,i);
		for(int i=-dsize; i<=dsize; i++)
			dm(k,i) /= sum;

	}
	
	for(int k=0; k < K;k++)
		for(int i=-dsize; i<=dsize; i++)
			for(int t=0; t<T; t++)
				d(t,k,i) = dm(k,i);
}

void importDict(double* d)
{
	for(int k=0; k < K;k++)
		for(int i=-dsize; i<=dsize; i++)
			dm(k,i) = d[(2*dsize+1)*k + (i+dsize) ];
	for(int k=0; k < K;k++)
	{
		double sum = 0;
		for(int i=-dsize; i<=dsize; i++)
			sum+=dm(k,i);
		for(int i=-dsize; i<=dsize; i++)
			dm(k,i) /= sum;
	}
	
	for(int k=0; k < K;k++)
		for(int i=-dsize; i<=dsize; i++)
			for(int t=0; t<T; t++)
				d(t,k,i) = dm(k,i);
}



void normalizeS(int t)
{
	double sum = 0;
	double sum2 = 0;
	for(int i = 0 ; i < ssize;i++)
	{
		sum+=s(t,i);
		sum2 =(s(t,i)*s(t,i))+sum2;
	} 
	double mean = sum/ssize;
	double std = sqrt(sum2/ssize-mean*mean);
	int anynan = 0;
	
	if(std == std && std > 0.0001)
	{
		for(int i = 0 ; i < ssize;i++)
		{
			s(t,i) = (s(t,i)-mean)/std;
			// s(t,i) = ((s(t,i)-mean));
			// if(s(t,i)!=s(t,i)) anynan++;
		}
	}
	// printf(">%d,%d,%f/%f/%f/%f/%d\n",t,ssize,std,sum,sum2,s(t,0),anynan);
}

void initS(int t, vector<double> in)
{
	vector<double> tmp(sigSize-bandFilterSize+1,0); 
	for(int i = 0 ; i < tmp.size(); i++)
	{
		tmp[i]=0;
		for(int j = 0 ; j < bandFilterSize; j++)
		{
			tmp[i] += in[i+j] * filter[j];
		}
	}
	
	for(int i = 0 ; i < ssize; i++)
	{
		s(t,i)= 0;
		for(int j = 0 ; j < MAFilterSize; j++)
		{
			s(t,i) += in[i+j] * MAFilter;
		}
	}
	normalizeS(t);
}

void initZ(int t) 
{
	for(int k=0;k<K;k++)
		for(int i = -dsize;i<ssize+dsize;i++)
			z(t,k,i) = 0;
}

void initBeta(int t)
{
	for(int k=0;k<K;k++)
	{
		for(int i = - dsize; i< ssize + dsize; i++)
		{
			b(t,k,i) = 0;
			for(int j =-dsize ; j <= dsize ;j++)
			{
				if(i+j <0 || i+j >=ssize) continue;
				b(t,k,i) += s(t,i+j) * d(t,k,-j);
			}
		}
	}
}

void normalizeDictionary(int t)
{
	for(int k = 0 ; k < K ; k++)
	{
		double total = 0.f;
		for(int i = -dsize; i <=dsize ; i++ )
			total += fabs(d(t,k,i));
		
		if(total != 0)
			for(int i = -dsize; i <=dsize ; i++ )
				d(t,k,i) /= total;
	}
}

int syncDictionary(int t)
{
	int isNan=0;
	for(int k = 0 ; k < K ; k++)
	{
		for(int i = -dsize; i <=dsize ; i++ )
		{
			
			if(d(t,k,i)!=d(t,k,i))isNan=1;
		}
	}

	if(isNan==0)
	{
		for(int k = 0 ; k < K ; k++)
		{
			for(int i = -dsize; i <=dsize ; i++ )
			{
				dm(k,i) = 0.9*dm(k,i) + 0.1*d(t,k,i);
			}
		}
	}
	else
	{
		printf("%d=NAN\n",t);
		// for(int i = 0 ; i < ssize; i++)
		// {
		// 	printf("%f ",s(t,i));
		// }

		// for(int k = 0 ; k < K ; k++)
		// {
		// 	for(int i = -dsize; i <=dsize ; i++ )
		// 	{
		// 		if(d(t,k,i)!=d(t,k,i))
		// 		{
		// 			printf("****%d %d %f\n",k,i,d(t,k,i));
		// 		}
		// 	}
		// }
	}
	
	for(int k = 0 ; k < K ; k++)
	{
		for(int i = -dsize; i <=dsize ; i++ )
		{
			d(t,k,i) = dm(k,i);
		}
	}
	return isNan;
}

int inference(int t)
{
	int round = 0;
	int lastK=-1;
	int lastI=-1;
	double loss = calcLoss(t);
	// printf("%d=LOSS=%f\n",t,loss);
	int negativeCount = 100;
	int positiveCount = 0;
	vector<int> lastDiff(100,0);

	while(true)
	{
		// calculate new Z
		for(int k=0;k<K;k++)
		{
			for(int i= -dsize ; i< ssize + dsize; i++)
			{
				double sqdivider = 0;
				for(int j = -dsize; j<=dsize; j++)
				{
					if(i+j < 0 || i+j >= ssize) continue;
					sqdivider += (d(t,k,-j) * d(t,k,-j));
				}
				zn(t,k,i) = shrink(b(t,k,i),LAMBDA) / sqdivider;
			}
		}
		//maximum diff Z
		double maxV=-1;
		int maxK;
		int maxI;

		for(int k =0 ; k < K ; k++)
		{
			for(int i=-dsize ; i < ssize + dsize; i++)
			{
				double fs = fabs(zn(t,k,i) - z(t,k,i) );
				if(fs > maxV )
				{
					if(lastK==k && lastI==i) printf("pick same \n");
					maxV=fs;
					maxK=k;
					maxI=i;
				}
			}
		}

		double savedBeta = b(t,maxK,maxI);
		// update beta
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
							total += d(t,maxK,c) * d(t,k,-(r-editing));
						}
					}
				}
				b(t,k,editing) -= (zn(t,maxK,maxI)-z(t,maxK,maxI)) * total;
			}
		}

		z(t,maxK,maxI) = zn(t,maxK,maxI);
		b(t,maxK,maxI) = savedBeta;
		double newLoss = calcLoss(t);
		double diffLoss = newLoss-loss;
		
		if(diffLoss>0)
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
			// if(fabs(loss-newLoss) < diffLossSPC) break;
			if(percentBreak > 0.4) break;
			if(round  > maxCoordinateDescentLoop ) break;
			if(newLoss==0) break;
		}
		loss = newLoss;
		round++;
	}
	// cout <<"]\n";
	return round;
}

int learnDictionary(int t)
{
	// double lastLoss= calcLoss(t);
	int round = 0 ;
	int special = 0;
	double dstepsize = dictstepsize;
	while(true)
	{
		vector<long double> precalc(ssize,0);
		for(int i = 0 ; i < ssize; i++)
		{
			precalc[i] = s(t,i);
			for(int k =0 ; k < K ;k++)
			{
				for(int j=-dsize;j<=dsize;j++)
				{
					precalc[i] -= d(t,k,j) * z(t,k,i+j);
				}
			}
		}
		for(int k=0;k<K;k++)
		{
			for(int d=-dsize;d<=dsize; d++)
			{
				nd(t,k,d) = 0;
				for(int i = 0 ; i < ssize;i++)
				{
					nd(t,k,d) += precalc[i] * -z(t,k,i+d);
				}
			}
		}
		for(int k=0;k<K;k++)
		{
			for(int ds=-dsize;ds<=dsize; ds++)
			{
				// if(nd(t,k,ds) > 1000) nd(t,k,ds) = 1000;
				// if(nd(t,k,ds) < -1000) nd(t,k,ds) = -1000;
				d(t,k,ds) -=  dstepsize * nd(t,k,ds);
			}
		}
		normalizeDictionary(t);

		round++;
		// printf("DLoss = %f %f = %f\n",lastLoss,loss,loss-lastLoss);
		// if(fabs(loss-lastLoss) < limitDiffLossDict) break;
		// lastLoss=loss;

		if(round > dictRound)
		{
			double loss = calcLoss(t);
			if(loss > 15)
			{
				special++;
				dstepsize = dictstepsize/5.;
				if(special == 120)
				{
					break;
				}
			}
			else
			{
				break;
			}
		}
		
	}
	return round;
}

void writeFile(int t,double loss)
{
	int count = 0;
	vector<float> val(0,0);
	vector<short> kindex(0,0);
	vector<short> iindex(0,0);
	
	for(int k =0 ; k < K ; k++)
	{
		for(int i=-dsize; i<ssize+dsize; i++)
		{
			if(fabs(z(t,k,i) ) > 0.0001)
			{
				kindex.push_back((short)k);
				iindex.push_back((short)i);
				val.push_back((float)z(t,k,i));
			}
		}
	}

	int recordLen = 8 + 2 + ssize*4  + 2 + (2+2+4) * val.size();
	
	outfile[t].write((char*) &recordLen, 4);
	outfile[t].write((char*) &loss, 8);
	short ssize_short = (short)ssize;
	outfile[t].write((char*) &ssize_short, 2);
	for(int i =0 ; i< ssize;i++)
	{
		float sf = (float)s(t,i);
		outfile[t].write((char*) &sf,4);
	}

	short zlen_short = (short)val.size();	
	outfile[t].write((char*) &zlen_short, 2);

	for(short i=0 ; i < zlen_short; i++)
	{
		outfile[t].write((char*) &kindex[i], 2);
		outfile[t].write((char*) &iindex[i], 2);
		outfile[t].write((char*) &val[i], 4);
	}
	outfile[t].flush();
}

void writeHead(int t)
{
	int recordLen = K*(dsize*2+1)*4 +4+4;
	outfile[t].write((char*) &recordLen, 4);
	outfile[t].write((char*) &dsize, 4);
	outfile[t].write((char*) &K, 4);
	
	for(int k =0 ; k < K ; k++)
	{
		for(int i=-dsize ; i <= dsize ; i++)
		{
			float dd = dm(k,i);
			outfile[t].write((char*) &dd,4);
		}
	}
	outfile[t].flush();
}

int main(int argc, char** argv)
{
	LAMBDA = atof(argv[1]);
	dsize = atoi(argv[2]);
	K = atoi(argv[3]);
	T = atoi(argv[4]);

	printf("lambda = %f\ndsize=%d\nK=%d\nT=%d\n",LAMBDA,dsize,K,T);
	printf("===================================\n");
	S = vector< vector<double> >(T,vector<double>(ssize+margin,0));
	OS = vector< vector<double> >(T,vector<double>(ssize+margin,0));
	D = vector< vector< vector<double> > >(T,vector< vector<double> >(K,vector<double>(dsize*2+1,0)));
	DMASTER = vector< vector<double> >(K,vector<double>(dsize*2+1,0));
	Z = vector< vector< vector<double> > >(T,vector< vector<double> >(K,vector<double>(ssize+3*margin,0)));
	B = vector< vector< vector<double> > >(T,vector< vector<double> >(K,vector<double>(ssize+3*margin,0)));

	Zct = vector< vector<double> >(T,vector<double>(K,0));
	ND = vector< vector< vector<double> > >(T,vector< vector<double> >(K,vector<double>(dsize*2+1,0)));
	Zn = vector< vector< vector<double> > > (T,vector< vector<double> >(K,vector<double>(ssize+3*margin,0)));

	vector< vector<double> > ts  = timeSeries("/home/kanit/npz/d-8634");
	vector<int> rd = shuffleArray(ts.size());
	printf("===================================\n");
	
	if(!flagImportDict)
	{
		initD();
	}
	else
	{
		double newdict[] = {0.06426783675,0.1219519919,0.1556892177,0.1633437298,0.1489696974,0.09221650604,0.04798539624,0.03049360724,0.03261175421,0.03614852888,0.03525843066,0.0282869851,0.01999634683,0.01412813644,0.008651834862,0.008050471485,0.02059276317,0.04004183772,0.06938398134,0.1064567833,0.1370707861,0.1509626029,0.1435438983,0.1197544448,0.08556121817,0.05289827559,0.03016104162,0.01867908234,0.01162775075,0.005215062464,0.0221694176,0.03757990562,0.03697086825,0.0004701392546,-0.07493494764,-0.1433736691,-0.1778722403,-0.169120858,-0.1175671087,-0.03393101621,0.03066121608,0.05536780233,0.04954194821,0.03308745044,0.01732563102,0.009069074853,0.01642994945,0.02309898782,0.02734194657,0.02934997263,0.03209524135,0.04218334074,0.06931387537,0.1121362173,0.1517393004,0.1670783405,0.145917064,0.1047540688,0.05544340143,0.01404921884,0.009580272999,0.0228118514,0.04253308533,0.07194266522,0.1071778,0.1360640921,0.1481366384,0.140470049,0.1174490369,0.0842872854,0.05228026975,0.0301397096,0.01902199985,0.01241296044,0.005692283482,0.003195799507,0.01315665703,0.03018053883,0.05731245292,0.09578968198,0.1297422425,0.1484361824,0.1448571512,0.1253557629,0.09334462504,0.06182862287,0.03907998006,0.02763965575,0.01977124368,0.01030940334,0.01940806364,0.03004843913,0.0403971074,0.04352667948,0.03319160009,0.01631120744,0.006344910396,0.01800122286,0.05604868827,0.1122438717,0.159911023,0.1726009572,0.146667732,0.0995336117,0.04576488563,0.003577169821,0.01123831472,0.02373495051,0.04654354804,0.0846218549,0.1250972139,0.151144272,0.154917583,0.1372170433,0.1029845356,0.06655613511,0.04019039253,0.02596674595,0.01726128808,0.008948952589,-0.01698589275,-0.0271207023,-0.02971108532,-0.02545076846,-0.01378778175,0.009634035814,0.04651298458,0.09329381388,0.1352607645,0.1610757162,0.1621008074,0.1356597857,0.08900685443,0.04272364321,0.01167536375,0.01073195531,0.0189894736,0.03346706616,0.06294418922,0.1159594759,0.1679388946,0.1784416039,0.1249477615,0.04897630421,-0.02317885711,-0.0737175886,-0.08202648887,-0.0389829145,0.003330965479,0.01636646111,0.000420064738,0.01192727464,0.03453850944,0.06030033632,0.09144552691,0.1212704634,0.1425330179,0.1473653677,0.1339311446,0.1069805784,0.0750470676,0.04531928036,0.02153464872,0.005028790477,-0.002357928817,0.03798632119,0.03532625655,0.03526202206,0.007914682877,0.03313946371,0.03836266211,0.1223350985,0.1467622193,0.1325179617,0.09541795053,0.05386782549,0.003043199063,0.03128590634,0.1135954976,0.1131829329,0.02620259561,0.0542215328,0.09339593395,0.1360601658,0.1688837774,0.1675402589,0.1299512813,0.07252057207,0.01741878498,-0.02025535949,-0.02351660205,-0.001753813792,0.02308565282,0.03518446863,0.03000920039,0.002883990739,0.01376277228,0.02796592264,0.0452248086,0.06762783257,0.09508149408,0.1260466984,0.1507210381,0.1559420762,0.1380697007,0.09706370285,0.04773227378,-0.01998061786,-0.006127876738,-0.005769194525,0.01268293052,0.04627907655,0.1078668933,0.1659441403,0.1748075213,0.1398130926,0.09837122771,0.04327082487,0.001918241311,0.004401423172,0.03604720566,0.05247870462,0.051257828,0.03983129808,0.02502959207,0.001511833947,0.009920940104,0.0250520415,0.04879679059,0.08623814953,0.123393669,0.1480207272,0.1514620006,0.1366690437,0.1046249462,0.07038971162,0.04327185958,0.02685045308,0.01635122174,0.007446611598,-0.01753377286,-0.02562026157,-0.03092536096,-0.03258109126,-0.02908363151,-0.02027358444,-0.004728903273,0.02735710412,0.08247420562,0.13585254,0.1663524643,0.167602096,0.1394838383,0.08569446462,0.03443668113,0.003461999084,0.01181130681,0.02520976273,0.04626647059,0.07848746696,0.1167825529,0.1452201224,0.1556162646,0.1412061084,0.1114079656,0.07452505664,0.04324798844,0.02458652136,0.01490065293,0.007269760517,-0.02271563755,-0.01182364376,0.02341170176,0.06587679961,0.08019216903,0.05143206336,0.007665113267,-0.006841695768,0.02058433267,0.0786349675,0.1484880019,0.1896832827,0.1635650243,0.0956649571,0.03342060965,-0.00139112078,0.02781938569,0.09797831215,0.1538532196,0.1814264659,0.1806537133,0.1386908365,0.07904343639,0.03976468233,0.02615290822,0.03048323168,0.03098498406,0.007630060594,-6.419216275e-05,-0.004063450542,0.01279332753,0.08825734168,0.05890705057,0.08569842012,0.1363182962,0.001379165251,0.06198350454,0.1110442613,0.02603661566,0.0522973368,0.1435501205,0.06469037179,0.03906638447,0.02829582753,0.08968197612,0.01456080265,0.017762308,0.02203277171,0.02861899509,0.03786977591,0.04080058153,0.03573167885,0.02770879663,0.03226742265,0.08041932373,0.1277880324,0.1594651935,0.1669644035,0.1416420297,0.06636788401,-0.004381148659,-0.01559164578,-0.04031747105,-0.08258441941,-0.1274149208,-0.1592315349,-0.1640414451,-0.1362757554,-0.07891790455,-0.01660974737,0.02949267611,0.05010884324,0.04781224488,0.03206018175,0.01516006095,0.0007616557507,0.009146417644,0.03291769273,0.07537354176,0.1303321916,0.1658821922,0.1410186696,0.05841037969,-0.0121440361,-0.03714057042,-0.02180219027,0.0270197971,0.09338697391,0.1181450414,0.07651864982,0.03672094155,0.03093657479,0.07684670646,0.08322595632,0.04535545109,0.08242134288,0.1254742918,0.05349879056,0.09830212795,0.02475295682,0.1404032373,-0.00789914112,0.04972150861,0.1024013017,0.04203967098,0.00349680245,0.01001593076,0.02220841898,0.04721689438,0.08657871738,0.1275486232,0.1539659823,0.1573448948,0.1373936323,0.1001920029,0.0632499439,0.03776843602,0.0251719029,0.01799161883,0.009856198924,0.003386220785,0.07257586141,0.1629448466,-0.04690967246,-0.08493755747,0.02019465019,0.01342520988,-0.09947448031,0.07850201487,0.1404735221,0.05021177438,0.01676505015,0.06499343254,0.08082461728,0.06438108956,-0.01428888006,-0.02233974541,-0.02962768343,-0.03536295049,-0.03824954935,-0.03817291437,-0.03088273558,-0.005836953541,0.04923186611,0.1136010786,0.1587701571,0.1708947547,0.1510277075,0.09968849094,0.04202453279,0.05617958095,0.01686954116,0.1124321245,0.1118055928,0.0429988462,0.1377539604,0.03712714672,-0.01629603794,0.04170336722,0.07889304353,0.05651804127,0.1377784417,0.1135762186,0.02686276863,0.01320528833,0.1088436321,0.1343307628,0.03572665147,-0.04354116725,-0.01878506091,0.01290486368,0.04917493661,0.1289334081,0.162675079,0.06066788153,-0.05785087953,-0.09674074499,-0.06411844058,-0.02514940694,0.0005527898777,0.01423451294,0.02277645246,0.0301975547,0.03432512906,0.03386683969,0.02583088309,0.00176897638,-0.05040612957,-0.113823242,-0.1594094934,-0.1742809651,-0.1591575661,-0.1119314074,-0.054600931,-0.01338991708,0.02580790182,0.09601311537,0.02216872926,0.04899928059,0.1364113003,0.02149428097,0.01699596853,0.08595399501,0.1154913055,-0.01189592781,0.1775861764,0.046474904,0.07491743967,0.05955528484,0.06023438984,0.02298493816,0.04062019744,0.05888360437,0.07468917347,0.08584605639,0.0917746244,0.09381610695,0.09301809672,0.0908836342,0.08684196193,0.07975313626,0.06857575129,0.05384974092,0.03735910171,0.02110387581,0.02730961175,0.05482166704,0.0842293241,0.09826960832,0.07525837175,0.006507178846,-0.07470843175,-0.1356024359,-0.1597187235,-0.1427333962,-0.08932358863,-0.03270647931,0.0009327063463,0.01014177382,0.007702502307,0.004130008159,0.01227539671,0.02622144828,0.04974388577,0.083474628,0.1189826765,0.1430573665,0.147413397,0.1335068464,0.1057527019,0.07320717003,0.04506113973,0.02807824499,0.01868328139,0.01041180859,0.02738367713,0.06508679347,0.1069305174,0.1387540428,0.1533938236,0.1454056378,0.1201997527,0.08743536708,0.05578779781,0.02811092812,0.005520890484,-0.01105471698,-0.01976376536,-0.02045170326,-0.01472058585,0.001285004418,0.009391754593,0.02539046426,0.0545304788,0.09695000724,0.1323176109,0.1507411037,0.1477278784,0.1277083313,0.09344278446,0.0626265251,0.04225127342,0.02946062945,0.01839056561,0.007785588433,0.002005918542,0.007936291047,0.02142866825,0.05118957486,0.09557547197,0.136529725,0.1578000408,0.1539935308,0.1293700552,0.08996266474,0.05613021253,0.03667108498,0.02834144723,0.0214862269,0.01157908718,-0.04076946485,-0.05631764326,-0.03502509095,0.01754615626,0.08004130679,0.1351501746,0.1637550567,0.1558571742,0.1219288071,0.083441385,0.05337898792,0.0327844603,0.01741857222,0.006472749309,0.0001093492773,0.01255668022,0.02148707644,0.02873714786,0.03379264558,0.03548709253,0.03323774613,0.02191946612,-0.01400199042,-0.0812307558,-0.1401866421,-0.1702595078,-0.1707202962,-0.1382782054,-0.07493788429,-0.02316686319,0.01062293686,0.02935185814,0.0686569427,0.1132954779,0.1497586632,0.164722467,0.1555341859,0.1193589528,0.07365107931,0.04003491546,0.03255637768,0.0248673105,0.00422018001,-0.006269442995,-0.007099209489,0.1125042235,0.00677472533,-0.003162616157,0.06326599059,0.03619343457,0.08278240816,0.08642830064,0.1714482427,0.1682807999,0.01956330258,-0.02218763421,0.08569762538,0.1218013842,-0.00240079502,-0.01750851702,0.02119894982,0.0529835882,0.02993249476,0.08803723502,0.08589423576,-0.000894596419,0.02580341306,0.02832873311,0.1542964701,0.05394399189,0.07837241596,0.09874097588,0.03396308515,-0.0225725373,0.2250372776,0.01011793946,0.02301442982,0.04505040082,0.08241887302,0.1240939265,0.1534571447,0.1613900283,0.147584391,0.1121473725,0.06991038916,0.03661406457,0.01866541106,0.01003075156,0.00237482664,-0.003130050794,0.005380124189,0.01436235469,0.02798393432,0.04756184833,0.0752791431,0.1095518678,0.1374906507,0.1475209669,0.1389291406,0.115310673,0.08154641322,0.04853297217,0.02680219466,0.01541349367,0.008334222736,0.02462589146,0.06674475765,0.1184683853,0.1574044754,0.1673043462,0.1475416676,0.1033925023,0.04955281974,0.0060604219,-0.01857213698,-0.02890372654,-0.03227868804,-0.03212817359,-0.02784720114,-0.01917480628,0.004605474947,0.01283637418,0.02872178381,0.05744158818,0.09650796296,0.1303497232,0.1474848024,0.1444119684,0.1250539065,0.09313298122,0.06256585857,0.04053156232,0.02785186332,0.01914861826,0.009355531799,0.002172581436,0.05126263934,0.03606774246,0.02036453418,0.1064732049,0.02616134051,0.1494386869,0.08910838445,0.05198950238,0.1463429026,0.04746823579,0.1102177758,-0.0007228016943,0.03127784735,0.1309318202,0.1092960238,0.004312876307,0.006139960115,0.08944966651,0.02020527997,0.09413144757,0.07088696156,0.07665798086,0.07262785046,0.08164346051,0.08928252023,0.01935678199,0.06978304279,0.1020466575,0.09417948983,-0.002071568293,0.0158329446,0.04600977482,0.08294927947,0.1144068467,0.1343654901,0.1359372461,0.1271037672,0.1091698374,0.08828195049,0.06671374915,0.04466364228,0.02348954931,0.00693857536,-0.002065778754};
		if(K*(2*dsize+1) != sizeof(newdict)/8)
		{
			printf("IMPORT DICT FAIL !!!!!!!");
			initD();
		}
		else
		{
			importDict(newdict);
		}
	}

	int count0 = 0;
	int countPrint = 0;
	int total = 0;
	int totalCode = 0;

	double lossRound =0;
	double zRound = 0;
	int totalRound = 0;
	
	outfile = new ofstream[T];
	for(int i = 0; i< T ; i++)
	{
		char outname[150];
		sprintf(outname,"%sdata-%d.bin",datapath.c_str(),i);
		outfile[i].open(outname,std::ofstream::binary);
		writeHead(i);
	}

	#pragma omp parallel for num_threads(T)
	for(int i = 0 ; i < ts.size() ; i++)
	{
		int t = omp_get_thread_num();
		
		initS(t,ts[rd[i]]);
		initZ(t);
		initBeta(t);
		double baseLoss = calcLoss(t);
		int inferRound = inference(t);
		double zLoss = calcLoss(t);
		#pragma omp critical(CRIT_1)
		{
			totalCode += inferRound;
			total++;
		
			lossRound +=zLoss;
			zRound +=inferRound;
			totalRound++;
			
			if(t==0)
			{
				count0++;
				if(count0 == miniBatch)
				{
					printf("*%d\t%d\tloss = %.5f \t-> %.5f(%8d) \t->%.5f\t->%.5f\n",t,total,baseLoss,zLoss,inferRound,(double)zRound/totalRound,lossRound/totalRound);
					count0=0;
					lossRound =0;
					zRound =0;
					totalRound=0;
				}
			}
			fflush(0);
		}
		writeFile(t,zLoss);
	}
 	for(int i = 0; i< T ; i++)
	{
		outfile[i].close();
	}

	return 0;
}