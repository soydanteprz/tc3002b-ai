import java.util.*;
import java.lang.*;
import java.io.*;
public class Main
{ 
  static  Reader sc=new Reader();
  // static FastReader sc=new FastReader(System.in);
  static class PairComparator implements Comparator<Pair>{
              
            // Overriding compare()method of Comparator 
                        // for descending order of cgpa
            
        }
 public static void main (String[] args) throws java.lang.Exception
   {
//	try{
	    /*
	        int n=sc.nextInt();
	        ArrayList<Integer> al=new ArrayList<>();
	        ArrayList<Long> al=new ArrayList<>();
	        Set<Integer> set=new HashSet<>();
	        Collections.sort(al,Collections.reverseOrder());
	     
	        long n=sc.nextLong();
	        for(int i=0;i<n;i++) 
	        String s=sc.next();
	        StringBuilder sb=new StringBuilder();
	     */
		  int t = sc.nextInt();
		  while(t-->0)
		    { 
		       int n=sc.nextInt();
		       int m=sc.nextInt();
		       int x=sc.nextInt();
		       PriorityQueue<Pair> pq=new PriorityQueue<>(m, new PairComparator());
		       for(int i=1;i<=m;i++)
		         {
		          Pair p=new Pair(0,i);
		          pq.add(p);
		         }
		       out.println("YES");
		       int a[]=new int[n];
		       for(int i=0;i<n;i++)
		         {
		           a[i]=sc.nextInt();
		           Pair shortest=pq.poll();
		           out.print(shortest.y+" ");
		           Pair update=new Pair(shortest.x+a[i],shortest.y);
		           pq.add(update);
		         }
		      
		      out.println( );
		    }
	     out.flush();
	     out.close();
// 	   }     
// 	catch(Exception e)
// 		 {}
	}
 
 /*
   ...SOLUTION ENDS HERE...........SOLUTION ENDS HERE...
 */
 

   
 /*                     
    Map<Long,Long> map=new HashMap<>();
             for(int i=0;i<n;i++)
               {
                  if(!map.containsKey(a[i]))
                   map.put(a[i],1);
                 else
                   map.replace(a[i],map.get(a[i])+1);
               }
     
    Set<Map.Entry<Long,Long>> hmap=map.entrySet();
             for(Map.Entry<Long,Long> data : hmap)
               {
          
               }
       
   Iterator<Integer> it = set.iterator();
          while(it.hasNext()) 
           { 
            int x=it.next();
           }
   */

static void print(int a[])
  {
     int n=a.length;
     for(int i=0;i<n;i++)
       {
          out.print(a[i]+" ");
       }
     out.println();
     out.flush();
  }
static void print(long a[])
  {
     int n=a.length;
     for(int i=0;i<n;i++)
       {
          out.print(a[i]+" ");
       }
     out.println();
     out.flush();
  } 


  


 

 
  
static class Pair
   {
      int x,y;
      Pair(int x,int y)
       {
          this.x=x;
          this.y=y;
       }
   }
   



 




 
 
 

   
   
    
  static  PrintWriter out=new PrintWriter(System.out);
  static int int_max=Integer.MAX_VALUE;
  static int int_min=Integer.MIN_VALUE;
  static long long_max=Long.MAX_VALUE;
  static long long_min=Long.MIN_VALUE;
}
