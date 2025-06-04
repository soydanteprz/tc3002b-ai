import java.io.*;
import java.util.*;


public class CODECHEF {
    
    static long MOD=1000000000;
    
    static long[] solve(int[] pos,long[] arr,int n,int k){
        long[] ans=new long[n];
        long[] left=new long[n];
        long[] right=new long[n];
        long min=Integer.MAX_VALUE;
        for(int i=0;i<n;i++){
            min=Math.min(min+1,arr[i]);
            left[i]=min;
        }
        min=Integer.MAX_VALUE;
        for(int i=n-1;i>=0;i--){
            min=Math.min(min+1,arr[i]);
            right[i]=min;
        }
        for(int i=0;i<n;i++){
            ans[i]=Math.min(left[i],right[i]);
        }
        return ans;


    }


    public static void main(String[] args) throws java.lang.Exception {
         FastReader fs=new FastReader(System.in);
//       StringBuilder sb=new StringBuilder();
//       PrintWriter out=new PrintWriter(System.out);
         int t=fs.nextInt();
         while (t-->0){
            int n=fs.nextInt();
            int k=fs.nextInt();
            int[] pos=new int[k];
            for(int i=0;i<k;i++)
                pos[i]=fs.nextInt()-1;
            long[] temp=new long[n];
            int ptr=0;
            Arrays.fill(temp,Integer.MAX_VALUE);
             for(int i=0;i<k;i++)
                 temp[pos[ptr++]]=fs.nextLong();
             long[] ans=solve(pos,temp,n,k);
             for(int i=0;i<n;i++)
                 System.out.print(ans[i]+" ");
             System.out.println();

         }
        //out.close;
    }
}
