import java.util.*;
				    import java.math.*;
				    import java.io.*; 
				    import java.util.concurrent.ThreadLocalRandom;
				    import java.util.stream.Collectors;

				     public class A{
				     
				  static InputStream inputStream = System.in;
		            static FastScanner scan=new FastScanner();
		          

				   
				     public static PrintWriter out = new PrintWriter (new BufferedOutputStream(System.out));
	 static boolean vis[]=new boolean[26];
	 static LinkedList<Integer>edges[];
	 static int cnt=0;
	 static ArrayList<Integer>arr=new ArrayList();
	 

			public static void main(String[] args)  throws Exception 
		      {
			
				//	scan=new FastScanner("lifeguards.in");
				//out = new PrintWriter("lifeguards.out");
				 
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
				 */


				    int tt=1;
				 
		

				     


	//System.out.println(2&0);
	/*for(int i=0;i<=70;i++)
		for(int j=0;j<=70;j++){
			if((i&j)==i)
			System.out.println(i+" "+j+" "+(i&j));
		}*/
		//System.out.println(1^14);
		//System.out.println(15&6);

	tt=scan.nextInt();
			 	    
			 outer:while(tt-->0)
				    {
				    	int l=scan.nextInt(),n=scan.nextInt();
				    	int arr[]=new int[n];
				    	long t[]=new long[n];
				    	int pos[]=new int[l+1];
				    	Arrays.fill(pos,-1);
				    	TreeSet<Integer>tree=new TreeSet<Integer>();
				    	for(int i=0;i<n;i++){
				    		arr[i]=scan.nextInt();
				    		tree.add(arr[i]);
				    		pos[arr[i]]=i;
				    	}
				    	
				    		for(int i=0;i<n;i++)
				    		t[i]=scan.nextLong();
				    	long L[]=new long[l+5];
				    	long R[]=new long[l+5];
				    	Arrays.fill(L,Integer.MAX_VALUE);
				    	Arrays.fill(R,Integer.MAX_VALUE);
				    	for(int i=1;i<=l;i++)
				    	{
				    		if(pos[i]!=-1)
				    		{
				    			L[i]=t[pos[i]];
				    		}
				    		L[i]=Math.min(L[i],L[i-1]+1);
				    	}
				    		for(int i=l;i>=1;i--)
				    	{
				    		if(pos[i]!=-1)
				    		{
				    			R[i]=t[pos[i]];
				    		}
				    		R[i]=Math.min(R[i],R[i+1]+1);
				    	}
				    	for(int i=1;i<=l;i++)
				    		out.print(Math.min(L[i],R[i])+" ");

out.println();
				    
				    
		           

		            }

				  	    out.close();
		 		  
				   
				}
				

				   
				     
				     
				    
				    
				    
				     
				    
				    
				     

				     
				    
				    }
