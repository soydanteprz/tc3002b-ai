import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigInteger;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.*;


public class Main {

	
	 
	
	 
	 
	
	 
	 
	 static Set<Long> set=new HashSet<>();
	 
	
	      
	
	
	
	
	
	
	
	
	
	
	 
	 
	 
	 
	 
	 
	
	public static void main(String[] args) throws IOException {
      FastReader sc=new FastReader();
      int t=sc.nextInt();
      while(t-->0) {
    	  int n=sc.nextInt();
    	  int m=sc.nextInt();
    	  int x=sc.nextInt();
    	  int a[]=new int[n];
    	  int b[]=new int[n];
    	  int freq[]=new int[100002];
    	  Arrays.fill(freq, 0);
    	  for(int i=0;i<n;i++) {
    		  a[i]=b[i]=sc.nextInt();
    	  }
    	  Arrays.sort(b);
    	  int temp=1;
   
    	  Map<Integer,List<Integer>> map=new HashMap<>();
    	  for(int i=0;i<n;i++) {
    		  if(map.get(b[i])==null) {
    			  List<Integer> lis=new ArrayList<Integer>();
    			  lis.add(temp);
    			  map.put(b[i], lis);
    		  }
    		  else {
    			  List<Integer> lis=map.get(b[i]);
    			  lis.add(temp);
    			  map.put(b[i], lis);
    		  }
    		  temp++;
    		  if(temp==m+1) {
    			  temp=1;
    		  }
    	  }
//    	  for(Map.Entry<Integer,List<Integer>> set: map.entrySet()) {
//    		   System.out.println(set.getKey()+" "+set.getValue());
//    	  }
    	  StringBuilder sb=new StringBuilder();
     	  List<Integer> res=new ArrayList<Integer>();
    	  for(int i=0;i<n;i++) {
    		  List<Integer> lis=map.get(a[i]);
    		  res.add(lis.get(freq[a[i]]));
    		//  sb.append(lis.get(freq[a[i]])+" ");
    		//  System.out.println(lis);
    		  freq[a[i]]++;
    	  }
    	  boolean flag=true;
    	  int arr[]=new int[m+1];
    	  for(int i=0;i<n;i++) {
    		  arr[res.get(i)] += a[i];
    	  }
    	  for(int i=1;i<=m-1;i++) {
    		
    			 long cal= Math.abs(arr[i+1]-arr[i]);
    			 if(cal >x) {
    				 flag=false;
    				 break;
    			 
    		 }
    		 if(!flag) break;
    	  }
    	  if(!flag) {
    		  System.out.println("NO");
    	  }
    	  else {
    		  System.out.println("YES");
    		  for(int i=0;i<res.size();i++) {
    			  System.out.print(res.get(i)+" ");
    		  }
    		  System.out.println();
    	  }
    	  
      }
	}

}
