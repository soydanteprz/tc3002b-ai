import java.util.*;
import java.io.*;

////***************************************************************************
 /* public class E_Gardener_and_Tree implements Runnable{

       public static void main(String[] args) throws Exception {
        new Thread(null, new E_Gardener_and_Tree(), "E_Gardener_and_Tree", 1<<28).start();
       }
    public void run(){
         WRITE YOUR CODE HERE!!!!
         JUST WRITE EVERYTHING HERE WHICH YOU WRITE IN MAIN!!!
       }

  }
*/
/////**************************************************************************


public class C_Set_or_Decrease{
    public static void main(String[] args) {
        FastScanner s= new FastScanner();
        //PrintWriter out=new PrintWriter(System.out);
        //end of program
        //out.println(answer);
        //out.close();
        StringBuilder res = new StringBuilder();
        int t=s.nextInt();
        int p=0;
        while(p<t){
        int n=s.nextInt();
        long k=s.nextLong();
        ArrayList<Long> list = new  ArrayList<Long>();
        long sum=0;
        for(int i=0;i<n;i++){
            long a=s.nextLong();
            list.add(a);
            sum+=a;
        }
        if(sum<=k){
            res.append("0 \n");
        }
        else{
            Collections.sort(list);
            long well[]= new long[n-1];
            long ss=0;
            for(int i=1;i<n;i++){
                long a=list.get(i);
                ss+=a;
                well[i-1]=ss;
              //  System.out.print(well[i-1]+" ");
            }
            System.out.println();
            long yo=k/n;
            long first=list.get(0);
            long yo1=Math.min(yo,first);
            long ans=Long.MAX_VALUE;
           // System.out.println("yo1 "+yo1);
            for(long i=yo1;i>(yo1-n-5);i--){
              //  System.out.println("i "+i);
                long num=first-i;
                long start=-1;
                long end=n-1;
                long mid=0;
                while(start<end){
                    mid=(start+end)/2;
                    if(start+1==end){
                        break;
                    }
                    long hh=well[(int)mid];
                    long a1=(n-mid-1);
                    long a2=(a1*i);
                    hh+=a2;
                    if(hh<=k){
                        start=mid;
                    }
                    else{
                        end=mid;
                    }
                }
                long jj=n-start-2;
                long kk=jj+num;
                ans=Math.min(ans,kk);



            }
            res.append(ans+" \n");
        }
        p++;
    }
    System.out.println(res);

    }
    


    

  
   // SIMPLE POWER FUNCTION=>


      





}
