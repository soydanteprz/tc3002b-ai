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


public class D_Blue_Red_Permutation{
    public static void main(String[] args) {
        FastScanner s= new FastScanner();
      //  PrintWriter out=new PrintWriter(System.out);
        //end of program
        //out.println(answer);
        //out.close();
        StringBuilder res = new StringBuilder();
        int t=s.nextInt();
        int p=0;
        while(p<t){
        int n=s.nextInt();
        long array[]= new long[n];
        for(int i=0;i<n;i++){
            array[i]=s.nextLong();
        }
        String str=s.nextToken();
        ArrayList<Long> red = new  ArrayList<Long>();
        ArrayList<Long> blue = new  ArrayList<Long>();
        for(int i=0;i<n;i++){
            if(str.charAt(i)=='R'){
                red.add(array[i]);
            }
            else{
                blue.add(array[i]);
            }
        }
        Collections.sort(blue);
        int check1=0;
        for(int i=0;i<blue.size();i++){
            int yo=i+1;
            if(blue.get(i)<yo){
                check1=1;
                break;
            }
        }
        Collections.sort(red,Collections.reverseOrder());
        int number=n;
        int check2=0;
        for(int i=0;i<red.size();i++){
             if(red.get(i)>number){
                 check2=1;
                 break;
             }
             number--;
        }
        if(check1==0 && check2==0){
            res.append("YES\n");
        }
        else{
            res.append("NO\n");
        }
        p++;
    }
    System.out.println(res);
    }
    




}
