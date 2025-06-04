import java.util.*;
import java.lang.*;
import java.io.*;
import java.math.BigInteger;
import java.text.DecimalFormat;

public final class Solution {

    static int inf = Integer.MAX_VALUE;
    static long mod = 1000000000 + 7;
  
 static void ne(Scanner sc, BufferedWriter op) throws Exception {
    int n=sc.nextInt();
    long k=sc.nextLong();
    long[] arr= new long[n];
    long sum=0;
    for(int i=0;i<n;i++){
        arr[i]=sc.nextLong();
        sum+=arr[i];
    }
    if(sum<=k){
        op.write("0\n");
        return;
    }
    sort(arr);
    long min=sum-k;
    long ss=0;
    
    int cc=0;
    for(int i=n-1;i>=1;i--){
          sum-=arr[i];
          sum+=arr[0];
          cc++;
        if(sum<=k){
          min=Math.min(min,cc);
          break;
        }
    }
   long val=Math.abs(arr[0]-(k/n))+(n-1);
    min=Math.min(val,min);
 
    for(int i=1;i<n;i++){
        ss+=arr[i];
        long no=((k-ss)/(long)(n-i));
     
        if(k-ss<0 && (k-ss)%(n-i)!=0){
            no--;
        }
       long count=(n-i-1)+Math.abs(arr[0]-no);
     
        min=Math.min(count,min);
    }
    op.write(min+"\n");




 }
 
  
    
  

    public static void main(String[] args) throws Exception {
        BufferedWriter op = new BufferedWriter(new OutputStreamWriter(System.out));
        // Reader sc = new Reader();
     
        Scanner sc= new Scanner(System.in);
        int t = sc.nextInt();
        while (t-->0){ ne(sc, op); }
        
        // ne(sc,op);
           
                   
        op.flush();
    }

    
    

    static void  sort(int[] arr){
        ArrayList<Integer> list= new ArrayList<>();
        for(int i=0;i<arr.length;i++){
            list.add(arr[i]);

        }
        Collections.sort(list);
        for(int i=0;i<arr.length;i++){
            arr[i]=list.get(i);
        }
    }
       static void  sort(long[] arr){
        ArrayList<Long> list= new ArrayList<>();
        for(int i=0;i<arr.length;i++){
            list.add(arr[i]);

        }
        Collections.sort(list);
        for(int i=0;i<arr.length;i++){
            arr[i]=list.get(i);
        }
    }

}

// return -1 to put no ahed in array










 
