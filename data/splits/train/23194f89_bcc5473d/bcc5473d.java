import java.util.*;
import java.lang.*;
import java.io.*;
import java.math.BigInteger;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.math.BigInteger;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;

public class k
{		//public static int mod=1000000007;
	
	 
	
	 
	    // method to return LCM of two numbers
	    
	 public static HashMap<Integer, Integer> sortByValue(HashMap<Integer, Integer> hm)
	    {
	        // Create a list from elements of HashMap
	        List<Map.Entry<Integer, Integer> > list =
	               new LinkedList<Map.Entry<Integer, Integer> >(hm.entrySet());
	 
	        // Sort the list
	        Collections.sort(list, new Comparator<Map.Entry<Integer, Integer> >() {
	            
	        });
	         
	        // put data from sorted list to hashmap
	        HashMap<Integer, Integer> temp = new LinkedHashMap<Integer, Integer>();
	        for (Map.Entry<Integer, Integer> aa : list) {
	            temp.put(aa.getKey(), aa.getValue());
	        }
	        return temp;
	    }
	 static int sieve =  1000000 ;


	 static boolean[] prime =  new boolean[sieve + 1] ;

	  
	 
	 
	 
	 
	 
	
	//if present - return the first occurrence of the no
	//not present- return the index of next greater value 
	//if greater than all the values return N(taking high=N-1)
	//if smaller than all the values return 0(taking low =0)
	
	//if present - return the index of next greater value
	//not present- return the index of next greater value 
	//if greater than all the values return N(taking high=N-1)
	//if smaller than all the values return 0(taking low =0)\
	
	
	
	
	 
	
	 public static void sortbyColumn(int arr[][], int col)  // send 2d array and col no
	    {
	        Arrays.sort(arr, new Comparator<int[]>() {       
	          
	        });
	    }
	 public static void sortbyColumn1(int arr[][], int col)  // send 2d array and col no
	    {
	        Arrays.sort(arr, new Comparator<int[]>() {       
	          
	        });
	    }
	 
	 
	    
	    
	    
	  
	  
	  
	 
	
	 
 	public static void main(String args[]) throws NumberFormatException, IOException ,java.lang.Exception
	{
		
		Reader reader = new Reader();
 	
			
		//sieveOfEratosthenes();
		//Scanner reader=new Scanner(System.in);
		PrintWriter out = new PrintWriter(System.out);
		BufferedWriter output = new BufferedWriter(new OutputStreamWriter(System.out));
//		BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
//		int cases=Integer.parseInt(br.readLine());
//		int cases=1;
		int cases=reader.nextInt();
		
   	while (cases-->0){
//   		long N=reader.nextLong();
   		//long M=reader.nextLong();
   		int N=reader.nextInt();
//   		int P=reader.nextInt();
	
   		
////   		
//   		String[] first=br.readLine().split(" ");
//   		long N=Long.parseLong(first[0]);
//   		long K=Long.parseLong(first[1]);
//   		long X=Long.parseLong(first[2]);
//   		String s2=br.readLine();
//   		String s3=br.readLine();
//   		char[]  s11=s2.toCharArray();
//	  	    char[]  s12=new char[s11.length];
   		
    	//int max=Integer.MIN_VALUE;
//   		int min=Integer.MAX_VALUE;
   		//int mod=1000000007;
 
//   		HashMap<Integer, TreeSet<Integer>> map=new HashMap<Integer,TreeSet<Integer>>();
   		
   		HashMap<Integer,Integer> map=new HashMap<Integer,Integer>();
//		HashMap<Long,Long> map=new HashMap<Long,Long>();
//   		HashMap<String,Integer> map1=new HashMap<String,Integer>();
   		//HashMap<Character,Integer> path=new HashMap<Character,Integer>();
   		
   		//TreeMap<Integer,Integer> map=new TreeMap<Integer,Integer>(Collections.reverseOrder());
  		
//   		HashSet<Integer> set =new HashSet<Integer>();
//   		HashSet<Integer> right =new HashSet<Integer>();

   		//   		TreeSet<Integer> a =new TreeSet<Integer>();
   		//TreeSet<Long> b =new TreeSet<Long>();
//   		TreeSet<Integer> map=new TreeSet<Integer>();
   		
//   		long[] arr=new long[N];
//   		int[] odd=new int[100001];
   		
   		int[] sec=new int[N];
   		int[] pow=new int[N];
//   		int[][] arr=new int[N][P];
   		ArrayList<Long> list=new ArrayList<Long>();
//   		ArrayList<Long> list3=new ArrayList<Long>();
//   		ArrayList<Long> list1=new ArrayList<Long>();
//   		ArrayList<Long> bees=new ArrayList<Long>();
   		
//   		boolean[]arr1=new boolean[N];
//   
//   		for(int i=0;i<N;i++)
//   		{
//   	
   		
   		for(int i=0;i<N;i++)
   		
   		{
   			sec[i]=reader.nextInt();
   		}
   		for(int i=0;i<N;i++)
   		{
   			pow[i]=reader.nextInt();
   		}
   		int last=sec[N-1];
   		int ind=sec[N-1]-pow[N-1]+1;
   		long ans=0;
   		int i=N-2;
   		while(i>=0)
   		{
//   			System.out.println(last +" "+ind);
   			if(sec[i]>=ind)
   			{
   				if(sec[i]-ind+1<pow[i])
   				{
   					ind=ind-(pow[i]-sec[i]+ind-1);
   				}
   				i--;
   			}
   			else if(sec[i]<ind)
   			{
   				ans=ans+((long)(last-ind+1))*((long)(last-ind+2))/(long)(2);
   				last=sec[i];
   		   		ind=sec[i]-pow[i]+1;
   		   		i--;
   			}
   			
   				
   		}
   		ans=ans+((long)(last-ind+1))*((long)(last-ind+2))/(long)(2);
   		System.out.println(ans);
   		
   	}
//  output.flush(); 
   	
	}
 
}
