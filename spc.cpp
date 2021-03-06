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

#define margin 		100 // becareful margin must me at least 3*dsize
#define os(t,i) 	OS[t][i+margin]
#define s(t,i) 		S[t][i+margin]
#define d(t,k,i) 	D[t][k][i+dsize]
#define dm(k,i) 	DMASTER[k][i+dsize]
#define nd(t,k,i) 	ND[t][k][i+dsize]
#define z(t,k,i) 	Z[t][k][i+margin]
#define zn(t,k,i) 	Zn[t][k][i+margin]
#define b(t,k,i) 	B[t][k][i+margin]
#define LAMBDA		0.000//0.5
#define MAXDICITER 	1

#define T 1
#define K 100
#define dsize 7
#define ssize 1250
#define importDict false
using namespace std;

string resultpath = "/home/kanit/anomalydeep/result/";

double minLossDict = 0.001;
double limitDiffLossDict = 0.001;
long double dictstepsize = 0.00005;

int inferenceMinRound = 4000;
int maxCoordinateDescentLoop = 6000;
double diffLossSPC = 0.0001;
int miniBatch = 1;
int maxData = 10000;
double impDict[K*(dsize*2+1)];

using namespace std;

double shrink(double beta, double lamb)
{
	if(beta>lamb) return beta - lamb;
	if(beta<-lamb) return beta + lamb;
	return 0;
}

vector< vector<double> > S(T,vector<double>(ssize+margin,0));
vector< vector<double> > OS(T,vector<double>(ssize+margin,0));
vector< vector< vector<double> > > D(T,vector< vector<double> >(K,vector<double>(dsize*2+1,0)));
vector< vector<double> > DMASTER(K,vector<double>(dsize*2+1,0));
vector< vector<double> > Zct(T,vector<double>(K,0));
vector< vector< vector<double> > > ND(T,vector< vector<double> >(K,vector<double>(dsize*2+1,0)));
vector< vector< vector<double> > > Z(T,vector< vector<double> >(K,vector<double>(ssize+3*margin,0)));
vector< vector< vector<double> > > Zn(T,vector< vector<double> >(K,vector<double>(ssize+3*margin,0)));
vector< vector< vector<double> > > B(T,vector< vector<double> >(K,vector<double>(ssize+3*margin,0)));
vector< vector<double> > ts;

double accumuZct(int t)
{
	for(int k=0;k<K;k++)
	{
		double ct = 0.f;
		for(int i = -dsize; i<ssize+dsize ;i++)
		{
			ct += fabs(z(t,k,i));
		}
		Zct[t][k] = (0.99)*Zct[t][k] + (0.01) * ct;
		cout << "Zct["<<k<<"]" << Zct[t][k]<<endl;
	}
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

void reportDZ(int t)
{
	vector<double> res = reconstruct(t);
	for(int i =0; i< ssize;i++)
	{
		cout << res[i] <<"\t";
	}
	cout<<endl;
}

void reportTesting(int t, string name,int infround)
{
	ofstream myfile;
	myfile.open (resultpath + name +".txt");
	
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

double lossDist(int t)
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
		cout<< fabs(x)<<"\t";
	}
//	if2
	return loss;
}

void reportZ(int t)
{
	for(int k = 0 ; k < K ; k++)
	{
		cout << "Z"<<k<<":\t";
		for(int i = - dsize; i<= ssize + dsize ;i++)
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

void reportS(int t) {
	//init signal
//	cout << "Signal("<< ssize <<") : ";
//	for(int i = 0 ; i < ssize;i++)s(i) = in[i];
	for(int i = 0; i < ssize;i++) cout << s(t,i) <<"\t";
	cout<<endl<<endl;
}

void initS(int t, vector<double> in) {
	//init signal
//	cout << "Signal("<< ssize <<") : ";
	for(int i = 0 ; i < ssize;i++)
	{
		s(t,i) = in[i];
		os(t,i) = in[i];
	}

//	for(int i = -5 ; i < ssize+5 ;i++) cout << s(i) <<" ";
//	cout<<endl<<endl;
}

int validateS(int t)
{
	for(int i = 0 ; i < ssize;i++)
	{
		if(s(t,i) != s(t,i)) return 0;
	}
	return 1;
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
	if(std ==0) return;
	for(int i = 0 ; i < ssize;i++)
	{
		// s(t,i) = ((s(t,i)-mean)/std);
		s(t,i) = ((s(t,i)-mean));
	}
}

void initZ(int t) {
	//init Z
	for(int k=0;k<K;k++)
	{
		for(int i = -dsize;i<ssize+dsize;i++)
		{
			z(t,k,i) = 0;
		}
	}
}
void syncDictionary(int t)
{
	int anyNan = 0;
	for(int k = 0 ; k < K ; k++)
	{
		for(int i = -dsize; i <=dsize ; i++ )
		{
			if(d(t,k,i) != d(t,k,i)) anyNan++;
		}
	}
	if(anyNan == 0)
	{
		for(int k = 0 ; k < K ; k++)
		{
			for(int i = -dsize; i <=dsize ; i++ )
			{
		
				dm(k,i) = 0.9*dm(k,i) + 0.1*d(t,k,i);
				d(t,k,i) = dm(k,i);
			}
		}
	}
	else
	{
		printf("NAN\n");
		for(int k = 0 ; k < K ; k++)
		{
			for(int i = -dsize; i <=dsize ; i++ )
			{
				for(int t = 0; t < T ; t++ )
				{
					d(t,k,i) = dm(k,i);
				}
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
		{
			total += d(t,k,i);
		}
		if(total != 0)
		{
			for(int i = -dsize; i <=dsize ; i++ )
			{
				d(t,k,i) /= total;
			}
		}
	}
}

void importDictionary()
{
	for(int k=0; k < K;k++)
	{
		for(int i=-dsize; i<=dsize; i++)
		{
			for(int t=0; t<T; t++)
			{
				d(t,k,i) = impDict[k*(dsize*2+1) + i+dsize];
			}
			dm(k,i) = impDict[k*(dsize*2+1) + i+dsize];
			printf("%f\t",d(0,k,i));
		}
	}
}

void initD()
{
	
	for(int k=0; k < K;k++)
	{
		for(int i=-dsize; i<=dsize; i++)
		{
			// d(0,k,i) = 1./(1+2*dsize) + 1./(1+2*dsize)*(rand()%10000)/1000000.0;
			// d(0,k,i) = 1./(1+2*dsize)*(rand()%10000)/1000000.0;
			d(0,k,i) = (rand()%1000)/1000000.0;
		}
	}
	
	// normalizeDictionary(0);
	
	// duplicate them to all thread
	for(int k=0; k < K;k++)
	{
		for(int i=-dsize; i<=dsize; i++)
		{
			for(int t=1; t<T; t++)
			{	
				d(t,k,i) = d(0,k,i);
			}
			dm(k,i) = d(0,k,i);
		}
	}
	
	// for(int k=0; k < K;k++)
	// {
	// 	cout << "D"<<k<<":\t";
	// 	for(int i=-dsize; i<=dsize; i++)
	// 	{
	// 		cout << d(0,k,i)<<"\t";
	// 	}
	// 	cout << endl;
	// }
	// cout << endl;
}

void initBeta(int t)
{
	// init Beta = D*z
	for(int k=0;k<K;k++)
	{
		for(int i = - dsize; i< ssize + dsize; i++)
		{
			b(t,k,i) = 0;
//			cout << "At : "<< i << " \n ";
			for(int j =-dsize ; j <= dsize ;j++)
			{
				if(i+j <0 || i+j >=ssize) continue;
//				cout << "s "<<i+j<<"\t d "<<-j <<endl;
				b(t,k,i) -= s(t,i+j) * d(t,k,-j);
			}
//			cout << b(k,i)<<endl;
		}
	}
}

int inference(int t)
{
	// Coordinate Descent
	int round = 0;
	double lastLoss = 0;
	int lastK=-1;
	int lastI=-1;
	char x;
	// cout << "L = [";
	while(true)
	{
		//calculate loss
		double loss =0 ;
		loss = calcLoss(t);
		printf("Loss : %f %f = %f\n",loss,lastLoss,loss-lastLoss);
		if(round%10==0)
		{
			// cout<<loss<<" ";
			// printf("%.5f\n",loss);
		}
		round++;

		if(round > inferenceMinRound )
		{
			// if(fabs(lastLoss-loss) < diffLossSPC)
			// {
			// 	// printf("end ! %.5f %d\n",loss,round);
			// 	break;
			// }
			if(round  > maxCoordinateDescentLoop ) break;
			if(loss==0) break;
		}
//		scanf("%c",&x);

//		reportZ();
//		reportD();
		lastLoss = loss;

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
				zn(t,k,i) = shrink(-b(t,k,i),LAMBDA) / sqdivider;
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
				if(fs > maxV  && lastK!=k && lastI!=i)
				{
					maxV=fs;
					maxK=k;
					maxI=i;
				}
			}
		}
//		cout << "maxV : " <<maxV ;
//		cout << "\tmaxK : " <<maxK ;
//		cout << "\tmaxI : " <<maxI ;
//		cout << "\tsig : " <<s(maxI) ;
//		cout << "\tnewZ : " << zn(maxK,maxI)<<endl;
//		reportB();
		// save old beta
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
//							cout << "sum : d("<<maxK<<","<<c<<")="<< d(maxK,c) <<" * d("<<k<<","<<-(r-editing)<<")"<<d(k,-(r-editing))<<"\n";
						}
					}
				}
				b(t,k,editing) += (zn(t,maxK,maxI)-z(t,maxK,maxI)) * total;
//				cout << "up : "<< k<<" "<<editing << " : " << b(k,editing) <<endl<<endl;
			}
		}

		// restore beta
//		if(round %10==0) cout << "change z : " << abs(z(maxK,maxI))<<endl;
		z(t,maxK,maxI) = zn(t,maxK,maxI);
		b(t,maxK,maxI) = savedBeta;
	}
	// cout <<"]\n";
	return round;
}

void learnDictionary(int t)
{
	long double lastLoss=99999999;
	int round = 0 ;
	while(round < MAXDICITER)
	{
		double loss = calcLoss(t);
		
		// cout << "DLoss("<<round<<") :\t";
		// printf("%.6f\n",loss);
		round++;

		if(fabs(loss-lastLoss) < limitDiffLossDict) return;
		lastLoss=loss;

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
				if(nd(t,k,ds) > 1000) nd(t,k,ds) = 1000;
				if(nd(t,k,ds) < -1000) nd(t,k,ds) = -1000;
				d(t,k,ds) -=  dictstepsize * nd(t,k,ds);
			}
		}
	}
}

// ^^^^^^^^^^^^^^^^^^^^^^ OPTIMIZATION PART
//-----------------------
// vvvvvvvvvvvvvvvvvvvvvv FILE INPUT PART

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
		if(maxData != 0 )
			countII = maxData;
		// while (!myReadFile.eof())
  //   	{
		// 	myReadFile.read((char*)&recordLength, 4);

		// 	myReadFile.read((char*)&sigName, 32);
		// 	myReadFile.read((char*)&sigStart, 8);
		// 	myReadFile.read((char*)&freq, 2);
		// 	myReadFile.read((char*)&sigLen, 2);
			
		// 	myReadFile.read((char*)&hasII, 1);
		// 	myReadFile.read((char*)&hasV, 1);
		// 	myReadFile.read((char*)&hasABP, 1);
		// 	myReadFile.read((char*)&unuse, 1);
			
		// 	myReadFile.read((char*)&baseII, 8);
		// 	myReadFile.read((char*)&gainII, 8);
		// 	myReadFile.read((char*)&baseV, 8);
		// 	myReadFile.read((char*)&gainV, 8);
		// 	myReadFile.read((char*)&baseABP, 8);
		// 	myReadFile.read((char*)&gainABP, 8);

		// 	// printf("Read Byte : %d\n",recordLength);
		// 	// printf("Name : %s\n",sigName);
		// 	// printf("SigStart : %ld\n",sigStart);
		// 	// printf("Frequency : %d\n",freq);
		// 	// printf("SigLength : %d\n",sigLen);
		// 	// printf("has II : %d\n",hasII);
		// 	// printf("has V : %d\n",hasV);
		// 	// printf("has ABP : %d\n",hasABP);
		// 	// printf("BASE GAIN : %f %f %f %f %f %f \n\n",baseII,gainII,baseV,gainV,baseABP,gainABP);

		// 	if(hasII==1) countII++;
		// 	if(hasV==1) countV++;
		// 	if(hasABP==1) countABP++;
			
		// 	if(hasII==1) myReadFile.read((char*)&shII, 1250*2);
		// 	if(hasV==1) myReadFile.read((char*)&shV, 1250*2);
		// 	if(hasABP==1) myReadFile.read((char*)&shABP, 1250*2);

		// 	if(freq != 125)
		// 	{
		// 		printf("***FREQ!=125\n");
		// 	}
		// 	// break;
		// }
		// printf("II %d / V %d / ABP %d\n",countII,countV,countABP);

		myReadFile.seekg(0, ios::beg);

		vector< vector<double> > v(countII,vector<double>(1250,0));

		int i = 0;
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

			// printf("Read Byte : %d\n",recordLength);
			// printf("Name : %s\n",sigName);
			// printf("SigStart : %ld\n",sigStart);
			// printf("Frequency : %d\n",freq);
			// printf("SigLength : %d\n",sigLen);
			// printf("has II : %d\n",hasII);
			// printf("has V : %d\n",hasV);
			// printf("has ABP : %d\n",hasABP);
			// printf("BASE GAIN : %f %f %f %f %f %f \n\n",baseII,gainII,baseV,gainV,baseABP,gainABP);

			if(hasII==1) myReadFile.read((char*)&shII, 1250*2);
			if(hasV==1) myReadFile.read((char*)&shV, 1250*2);
			if(hasABP==1) myReadFile.read((char*)&shABP, 1250*2);

			if(hasII==1)
			{
				for(int j=0; j< 1250;j++)
				{
					v[i][j] = (shII[j] - baseII) * gainII;
				}
				i++;
			}
			// break;
			if(i==countII) break;
		}
		printf("read file done");
		return v;
	}
	else
	{
		cout << "404 file not found ! :(" << endl;
		vector< vector<double> > v(0,vector<double>(0,0));
		return v;
	}
}

vector<int> shuffleArray(int total)
{
	vector<int> v(total,0);
	for(int i = 0 ; i < total ; i++)
	{
		v[i]=i;
	}
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(v.begin(), v.end(), g);
    return v;
}

int main(void){
	ts = timeSeries("/home/kanit/npz/d-8634");

	// for(int i = 0; i < 1250;i++)
	// {
	// 	printf("%.4f,",ts[93939][i]);
	// }
	// return 0;
//	cout << ts.size()<<endl; return 0;
//	vector< vector<double> > ts(1,vector<double>(1250,0));
//	for(int j=0;j<ssize;j++) ts[0][j]=tss[j];

	if(importDict)
	{
		importDictionary();
	}
	else
	{
		initD();
	}

	double sumLoss = 0;
	int countLoss = 0;
	int sumRound = 0;
	int count0 = 0;
	int countPrint = 0;
	int count = 0;
	vector<int> rd = shuffleArray(ts.size());

	#pragma omp parallel for num_threads(T)
	for(int i = 0 ; i < ts.size() ; i++)
	{
		int t = omp_get_thread_num();
		
		initS(t,ts[6134]);
		initZ(t);
		normalizeS(t);
		initBeta(t);
		int inferRound = inference(t);
		double loss = calcLoss(t);
		learnDictionary(t);
		// normalizeDictionary(t);
		// printf("-");
		fflush(0);
		
		#pragma omp critical(CRIT_1)
		{
			sumLoss += loss;
			countLoss++;
			sumRound += inferRound;
			count++;
			if(t==0) count0++;
			if(t==0 && count0 == miniBatch)
			{
				char logname[100];
				sprintf(logname,"res_%06d_%d",countPrint++,(int)(sumLoss/countLoss));
				reportTesting(t,logname,inferRound);
				printf("AvgLoss (%d - %d): %f (%d)\n",count,countLoss,sumLoss/countLoss,sumRound/countLoss);
				countLoss = 0 ;
				count0 = 0 ;
				sumLoss = 0 ;
				sumRound = 0 ;
			}

			syncDictionary(t);
		}
	}
}

