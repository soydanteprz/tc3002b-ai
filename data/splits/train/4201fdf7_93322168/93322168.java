import java.util.*;
import java.lang.*;
import java.io.*;
public class Main
{ 
  static  Reader sc=new Reader();
  // static FastReader sc=new FastReader(System.in);
  static ArrayList<Integer> one;
  static ArrayList<Integer> zero;
  static long dp[][];
  static void ini()
   {
      for(long a[] : dp)
        Arrays.fill(a,-1);
   }
static long solve(int i,int j)
 {
    if(i==one.size()) return 0;
    if(j==zero.size()) return int_max;
    
    if(dp[i][j]!=-1) return dp[i][j];
    
     long ans1=Math.abs(one.get(i)-zero.get(j))+solve(i+1,j+1);
     long ans2=solve(i,j+1);
     return dp[i][j]=Math.min(ans1,ans2);
 }
   // static int ceilSearch(ArrayList<Integer> arr, int low, int high, int x)
   //  {
   //    int mid;   
       
   //    /* If x is smaller than or equal to the
   //       first element, then return the first element */
   //    if(x <= arr.get(low))
   //     return low;
      
   //    /* If x is greater than the last
   //       element, then return -1 */
   //    if(x > arr.get(high))
   //     return -1; 
      
   //    /* get the index of middle element
   //       of arr[low..high]*/
   //    mid = (low + high)/2;  /* low + (high - low)/2 */
      
   //    /* If x is same as middle element,
   //       then return mid */
   //    if(arr.get(mid) == x)
   //     return mid;
          
   //    /* If x is greater than arr[mid], then
   //       either arr[mid + 1] is ceiling of x or
   //       ceiling lies in arr[mid+1...high] */
   //    else if(arr.get(mid) < x)
   //    {
   //     if(mid + 1 <= high && x <= arr.get(mid+1))
   //        return mid + 1;
   //     else
   //        return ceilSearch(arr, mid+1, high, x);
   //    }
      
   //    /* If x is smaller than arr[mid],
   //       then either arr[mid] is ceiling of x
   //       or ceiling lies in arr[low...mid-1] */  
   //    else
   //    {
   //     if(mid - 1 >= low && x > arr.get(mid-1))
   //        return mid;
   //     else   
   //        return ceilSearch(arr, low, mid - 1, x);
   //    }
   //  }
    
   //  static int floorSearch(
   //     ArrayList<Integer> arr, int low,
   //     int high, int x)
   //  {
   //     // If low and high cross each other
   //     if (low > high)
   //          return -1;
 
   //     // If last element is smaller than x
   //     if (x >= arr.get(high))
   //          return high;
 
   //     // Find the middle point
   //     int mid = (low + high) / 2;
 
   //     // If middle point is floor.
   //     if (arr.get(mid) == x)
   //          return mid;
 
   //     // If x lies between mid-1 and mid
   //     if (
   //          mid > 0 && arr.get(mid - 1) <= x && x < arr.get(mid))
   //          return mid - 1;
 
   //     // If x is smaller than mid, floor
   //     // must be in left half.
   //     if (x < arr.get(mid))
   //          return floorSearch(
   //              arr, low,
   //              mid - 1, x);
 
   //     // If mid-1 is not floor and x is
   //     // greater than arr[mid],
   //     return floorSearch(
   //          arr, mid + 1, high,
   //          x);
   //  }
 
 
 public static void main (String[] args) throws java.lang.Exception
   {
	try{
	  
		  int t = 1;// sc.nextInt();
		  int max=5001;
		  dp=new long[max][max];
		  ini();
		  one=new ArrayList<>();
		  zero=new ArrayList<>();
		  while(t-->0)
		    { 
		       int n=sc.nextInt();
		       long ans=0;
		       int a[]=new int[n];
		       for(int i=0;i<n;i++)
		         {
		           a[i]=sc.nextInt();
		           if(a[i]==1)
		            one.add(i);
		           else
		            zero.add(i);
		         }
		    //  print_int(one);
		    //  print_int(zero);
		      
		    //  for(int i=0;i<one.size();i++)
		    //    {
		    //       int ele=one.get(i);
		    //       int floor=floorSearch(zero,0,zero.size()-1,ele);
		    //       int ceil=ceilSearch(zero,0,zero.size()-1,ele);
		           
		    //       int left=int_max;
		    //       if(floor!=-1)
		    //        left=zero.get(floor);
		           
		    //       int right=int_max;
		    //       if(ceil!=-1)
		    //        right=zero.get(ceil);
		          
		    //      int left_val=Math.abs(ele-left);
		    //      int right_val=Math.abs(ele-right);
		          
		    //      if(left_val<=right_val)
		    //        {
		    //           ans+=left_val;
		    //           zero.remove(floor);
		    //        }
		    //      else
		    //        {
		    //           ans+=right_val;
		    //           zero.remove(ceil);
		    //        }
		    //    }
		    //   out.println(ans);
		    
		    ans=solve(0,0);
		    out.println(ans);
		    }
	     out.flush();
	     out.close();
	   }     
	catch(Exception e)
		 {}
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


 




 
  
// static class Pair
//    {
//       int x,y;
//       Pair(int x,int y)
//        {
//           this.x=x;
//           this.y=y;
//        }
//    }
   



 




 
 
 

   
   
    
  static  PrintWriter out=new PrintWriter(System.out);
  static int int_max=Integer.MAX_VALUE;
  static int int_min=Integer.MIN_VALUE;
  static long long_max=Long.MAX_VALUE;
  static long long_min=Long.MIN_VALUE;
}
