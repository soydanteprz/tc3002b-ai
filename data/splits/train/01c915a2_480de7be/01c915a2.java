import java.util.*;
					    import java.math.*;
					    import java.io.*; 
					    import java.util.stream.Collectors;

					       public class A{
					     
					  static InputStream inputStream = System.in;
			            static FastScanner scan=new FastScanner();
					     public static PrintWriter out = new PrintWriter (new BufferedOutputStream(System.out));
	static final long p=29;
	static final long mod=(long)1e9+9;
	static int n,m;
	static String spotty[],plain[];
	

				public static void main(String[] args)  throws Exception 
			      {
				
						///scan=new FastScanner("cownomics.in");
					//out = new PrintWriter("cownomics.out");
					 
					 /*
					 currently doing 
					 1-digit dp
					 2-ds like fenwick and interval tree and sparse table 
					 */
					 /*
					 READING 
					 1-Everything About Dynamic Programming codeforces
					 2-DYNAMIC PROGRAMMING: FROM NOVICE TO ADVANCED topcoder
					 3-Introduction to DP with Bitmasking codefoces
					 4-Bit Manipulation hackerearth
					 5-read more about mobious and inculsion-exclusion 
					 */
		



					    int tt=1;
					tt=scan.nextInt();
					int T=tt;
					/*Map<Integer,Pair>map=new HashMap<Integer,Pair>();
					for(int i=1;i<=333333334;i++)
					{
						map.put(i+2*i,new Pair(i,2*i));
						map.put(i+(2*(i+1)),new Pair(i,i+1));
						map.put(i+1+(2*i),new Pair(i+1,i));
					}*/

				 outer:while(tt-->0)
					    {
					    int n=scan.nextInt();
					   int cnt[][]=new int[n][5];
					   String arr[]=new String[n];
					   int sumlens=0;
					   for(int i=0;i<n;i++)
					   {
					   	String s=scan.next();
					   	sumlens+=s.length();
					   	arr[i]=s;
					   	for(int j=0;j<s.length();j++)
					   		cnt[i][s.charAt(j)-'a']++;
					   }
					   int res=0;
					   for(int to=0;to<5;to++)
					   {
					   	ArrayList<Pair>tmp=new ArrayList<Pair>();
					   	int sumall=0;
					   	for(int i=0;i<n;i++)
					   		sumall+=cnt[i][to];
					   	for(int i=0;i<n;i++)
					   	{
					   		int sum=0;
					   		for(int j=0;j<5;j++)
					   		{
					   			if(j!=to)
					   			{
					   				sum+=cnt[i][j];
					   			}
					   		}
					   		tmp.add(new Pair(arr[i].length()-sum,sum));
					   	}
					   	Collections.sort(tmp);
					   	int THIS=sumall,THAT=sumlens-sumall;
					   	//if(to==3)
					  // 		out.println(tmp);
					   	for(int i=0;i<tmp.size();i++)
					   	{
					   		if(THIS>THAT)
					   		{
					   		//	if(to==3)
					   		//		out.println("AHA "+(n-i));
					   			res=Math.max(res,n-i);
					   			break;
					   		}
					   		THIS-=tmp.get(i).x;
					   		THAT-=tmp.get(i).y;
					   	}
					   }
					   out.println(res);
					   
					    

					   

					   
					    	
					    	
			}
			   out.close();
		}
		  	 
			 		  
					   
					
					

					   
					     
					     
					    
					    
					    
					     
					    private static void sort(long[] arr) {
					            List<Long> list = new ArrayList<>();
					            for (long object : arr) list.add(object);
					            Collections.sort(list);
					        //Collections.reverse(list);
					            for (int i = 0; i < list.size(); ++i) arr[i] = list.get(i);
					        }
					    
					     

					     static class Pair implements Comparable<Pair>{
					            public long x, y,z;
					            public Pair(long x1, long y1,long z1) {
					                x=x1;
					                y=y1;
					                z=z1;
					            }
					             public Pair(long x1, long y1) {
					                x=x1;
					                y=y1;
					    
					            }
					            
					            
					            public String toString() {
					                return x + " " + y+" "+z;
					            }
					            
					    
					     
					    }
					    
					    }
