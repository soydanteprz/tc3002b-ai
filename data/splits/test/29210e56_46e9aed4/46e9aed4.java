import java.util.*;

import java.lang.*;
import java.io.*;

/* Name of the class has to be "Main" only if the class is public. */
public final class Solution {
    

    public static void main(String[] args) throws Exception {
        
        Reader sc = new Reader();
        BufferedWriter op = new BufferedWriter(new OutputStreamWriter(System.out));
         
         int n=sc.nextInt();
          ArrayList<Integer> fill= new ArrayList<Integer>();
          ArrayList<Integer> unfilled= new ArrayList<>();
          for(int i=0;i<n;i++){
              int x =sc.nextInt();
              if(x==1){
                  fill.add(i);
              }else{
                  unfilled.add(i);
              }
          }
          Collections.sort(fill);
          Collections.sort(unfilled);
          
          
          long[][] dp =new long[fill.size()+1][unfilled.size()+1];
           for(int i=0;i<fill.size()+1;i++){
              for(int j=0;j<unfilled.size()+1;j++){
                  dp[i][j]=Integer.MAX_VALUE;
              }
          }
          for(int i=0;i<unfilled.size()+1;i++){
              dp[0][i]=0;
          }
        //   for(int j=0;j<fill.size()+1;j++){
        //       dp[j][0]=0;
        //   }
         

          for(int i=1;i<fill.size()+1;i++){
              for(int j=1;j<unfilled.size()+1;j++){
                  dp[i][j]=Math.min(dp[i][j-1],dp[i-1][j-1]+Math.abs(fill.get(i-1)-unfilled.get(j-1)));
              }
          }
          System.out.println(dp[fill.size()][unfilled.size()]);
        //   for(int i=0;i<fill.size()+1;i++){
        //       for(int j=0;j<unfilled.size()+1;j++)
        //       {
        //           System.out.print(dp[i][j]+" ");
        //       }
        //       System.out.println();
        //   }
         
    }




   
    }


