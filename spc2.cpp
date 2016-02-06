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

#define importDict false
#define maxData 2000000
#define sigSize 1250
#define bandFilterSize 47 //@ 2 hz
#define MAFilterSize 5 
#define MAFilter 0.20 

//TO TUNE
// lambda dictsize num_dict
double LAMBDA = 0.05;
int dsize = 4;
int T = 4;
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

vector< vector<double> > S;
vector< vector<double> > OS;
vector< vector< vector<double> > > D;
vector< vector<double> > DMASTER;
vector< vector< vector<double> > > Z;
vector< vector< vector<double> > > B;

vector< vector<double> > Zct;
vector< vector< vector<double> > > ND;
vector< vector< vector<double> > > Zn;


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
	initD();

	int count0 = 0;
	int countPrint = 0;
	int total = 0;
	int totalCode = 0;

	double lossRound =0;
	double zRound = 0;
	int totalRound = 0;
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
		int dictRound = learnDictionary(t);
		double dLoss = calcLoss(t);
		#pragma omp critical(CRIT_1)
		{
			totalCode += inferRound;
			total++;
			int isNaN = syncDictionary(t);
			double sLoss = calcLoss(t);

			lossRound +=sLoss;
			zRound +=inferRound;
			totalRound++;
			if(isNaN==1)
			{
				printf("NAN\n");
			}
			if(t==0 )
			{
				count0++;
				if(count0 == miniBatch)
				{
					char logname[100];
					sprintf(logname,"res_%06d_%d_%d_%d",countPrint++,(int)(dLoss),K,T);
					char foldname[100];
					sprintf(foldname,"res_lb%f_ds%d_k%d_t%d",LAMBDA,dsize,K,T);
					reportTesting(t,foldname, logname,inferRound);
					printf("*%d\t%d\tloss = %.5f \t-> %.5f(%8d) \t-> %.5f(%8d)\t-> %.5f\t->%.5f\t->%.5f\n",t,total,baseLoss,zLoss,inferRound,dLoss,dictRound,sLoss,(double)zRound/totalRound,lossRound);
					count0=0;
					lossRound =0;
					zRound =0;
					totalRound=0;
				}
				else
				{
					printf(" %d\t%d\tloss = %.5f \t-> %.5f(%8d) \t-> %.5f(%8d)\t-> %.5f\t->%.5f\n",t,total,baseLoss,zLoss,inferRound,dLoss,dictRound,sLoss,(double)totalCode/total);
				}
			}
		}
	}
 
	return 0;
}