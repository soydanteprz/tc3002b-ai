import java.util.*;

public class Solve{

    public static void main(String[] args){
        Scanner sc=new Scanner(System.in);
        int t=sc.nextInt();
       
       
        while(t-->0){
            int n=sc.nextInt();
           long ans=Long.MAX_VALUE;
           long pre=0;
           PriorityQueue<Long> epq=new PriorityQueue<>();
           PriorityQueue<Long> opq=new PriorityQueue<>();
           for(int i=0;i<n;i++){
               long a=sc.nextInt();
               if(i%2==0)opq.add(a);
               else epq.add(a);
               pre+=a;
              if(i>0) ans=Math.min(ans,pre+opq.peek()*(n-opq.size())+epq.peek()*(n-epq.size()));
               
           }
          System.out.println(ans);
        }
    }
}
